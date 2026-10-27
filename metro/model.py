"""PVCGN re-implemented in plain PyTorch (dense adjacency), plus the PT-PVCGN extensions.

Faithful to the reference code (reference_PVCGN/ggnn/multigraph.py, rgcn.py, config
*_global_local_fusion_input.yaml): relational graph convolution with basis decomposition
(K=1 hop, 3 bases), GC-GRU + FC-GRU fusion (CGRM), 2-layer seq2seq with curriculum learning.

Extensions for the research gap "PVCGN ignores periodic and calendar context"
(each switchable, for ablations; both off = PVCGN):
  periodic  (P): every encoder/decoder step also receives the same slot on the most recent
                 same-type day and on the weekly reference day.
  temporal  (T): time-of-day, day-of-week and working-day embeddings appended to node inputs.
"""
from __future__ import annotations

import math
import random

import torch
from torch import nn
from torch.nn import Parameter, init


def normalize_relations(adj: torch.Tensor) -> torch.Tensor:
    """[R, N, N] adjacency A[r, src, dst] -> aggregation matrices M[r, dst, src].

    Matches the reference: diagonal zeroed, each destination's incoming weights sum to 1
    per relation (``adj / adj.sum(axis=0)``), messages flow src -> dst.
    """
    adj = adj.clone()
    idx = torch.arange(adj.shape[-1])
    adj[:, idx, idx] = 0
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-18)
    return adj.transpose(1, 2).contiguous()


class RelGraphConv(nn.Module):
    """out_i = x_i W_root + sum_r sum_j M_r[i, j] x_j W_r,   W_r = sum_b att[r, b] B_b."""

    def __init__(self, in_channels, out_channels, num_relations, num_bases):
        super().__init__()
        self.num_relations = num_relations
        self.basis = Parameter(torch.empty(num_bases, in_channels, out_channels))
        self.att = Parameter(torch.empty(num_relations, num_bases))
        self.root = Parameter(torch.empty(in_channels, out_channels))
        for p in (self.basis, self.att, self.root):
            init.xavier_uniform_(p)

    def forward(self, x, static_m):
        """x [B, N, C]; static_m [R, N, N]."""
        # aggregate every relation first, then one fused matmul: [x | M_1 x | ... | M_R x] @ [W_root; W_1; ...; W_R]
        w = torch.einsum("rb,bio->rio", self.att, self.basis)
        agg = torch.einsum("rij,bjc->bric", static_m, x)
        b, r, n, c = agg.shape
        feats = torch.cat([x, agg.permute(0, 2, 1, 3).reshape(b, n, r * c)], -1)
        return feats @ torch.cat([self.root, w.reshape(r * c, -1)], 0)


class CGRMCell(nn.Module):
    """GC-GRU with optional FC-GRU global fusion (the paper's CGRM)."""

    def __init__(self, in_channels, units, num_nodes, num_relations, num_bases=3, global_fusion=True):
        super().__init__()
        self.units = units
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion
        self.conv_i = RelGraphConv(in_channels, 3 * units, num_relations, num_bases)
        self.conv_h = RelGraphConv(units, 3 * units, num_relations, num_bases)
        self.bias_r = Parameter(torch.ones(units))
        self.bias_i = Parameter(torch.ones(units))
        self.bias_n = Parameter(torch.ones(units))
        if global_fusion:
            self.mlpi = nn.Linear(num_nodes * in_channels, units)
            self.mlph = nn.Linear(num_nodes * units, units)
            self.bias_r_g = Parameter(torch.ones(units))
            self.bias_i_g = Parameter(torch.ones(units))
            self.bias_n_g = Parameter(torch.ones(units))
            self.global_i = nn.Linear(units, 3 * units)
            self.global_h = nn.Linear(units, 3 * units)
            self.mlpatt = nn.Linear(2 * units, units)
            self.ln = nn.LayerNorm([units])

    def forward(self, x, h, static_m):
        b = x.shape[0]
        if h is None:
            h = x.new_zeros(b, self.num_nodes, self.units)
        i_r, i_i, i_n = self.conv_i(x, static_m).chunk(3, -1)
        h_r, h_i, h_n = self.conv_h(h, static_m).chunk(3, -1)
        r = torch.sigmoid(i_r + h_r + self.bias_r)
        z = torch.sigmoid(i_i + h_i + self.bias_i)
        n = torch.tanh(i_n + r * h_n + self.bias_n)
        nh = (1 - z) * n + z * h
        if self.global_fusion:
            gi = self.mlpi(x.reshape(b, -1))
            gh = self.mlph(h.reshape(b, -1))
            ir, ii, inn = self.global_i(gi).chunk(3, -1)
            hr, hi, hn = self.global_h(gh).chunk(3, -1)
            rg = torch.sigmoid(ir + hr + self.bias_r_g)
            zg = torch.sigmoid(ii + hi + self.bias_i_g)
            # the reference code adds bias_i_g here (not bias_n_g); kept for fidelity
            ng = torch.tanh(inn + rg * hn + self.bias_i_g)
            og = ((1 - zg) * ng + zg * gh).unsqueeze(1).expand(-1, self.num_nodes, -1)
            nh = nh * torch.tanh(self.mlpatt(torch.cat([nh, og], -1)))
            nh = self.ln(nh)
        return nh


class Net(nn.Module):
    def __init__(self, num_nodes, static_adj, units=256, num_layers=2, horizon=4,
                 periodic=False, temporal=False, cl_decay_steps=200,
                 use_curriculum_learning=True, global_fusion=True, slots_per_day=73):
        super().__init__()
        self.num_nodes, self.units, self.horizon = num_nodes, units, horizon
        self.periodic, self.temporal = periodic, temporal
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.register_buffer("static_m", normalize_relations(torch.as_tensor(static_adj, dtype=torch.float32)))
        num_rel = self.static_m.shape[0]
        if temporal:
            self.tod_emb = nn.Embedding(slots_per_day, 16)
            self.dow_emb = nn.Embedding(7, 4)
            self.wd_emb = nn.Embedding(2, 4)
        in_dim = 2 + (4 if periodic else 0) + (24 if temporal else 0)
        make = lambda i, bases=3: CGRMCell(i, units, num_nodes, num_rel, bases, global_fusion=global_fusion)
        self.encoder = nn.ModuleList([make(in_dim)] + [make(units) for _ in range(num_layers - 1)])
        # the reference passes K (=1) positionally as num_bases for the upper decoder layers; kept for fidelity
        self.decoder = nn.ModuleList([make(in_dim)] + [make(units, 1) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(units, 2)
        self.global_step = 0

    def _context(self, batch, step):
        """Extra node inputs (periodic references, calendar embeddings) for absolute step 0..7 of the window."""
        extra = []
        if self.periodic:
            extra += [batch["ref_type"][:, step], batch["ref_week"][:, step]]  # [B, N, 2] each
        if self.temporal:
            slot = batch["slots"][:, step]                               # [B]
            t = torch.cat([self.tod_emb(slot), self.dow_emb(batch["dow"]), self.wd_emb(batch["workday"])], -1)
            extra.append(t.unsqueeze(1).expand(-1, self.num_nodes, -1))
        return extra

    def _step(self, cells, x, hidden, extra):
        inp = torch.cat([x] + extra, -1) if extra else x
        for i, cell in enumerate(cells):
            hidden[i] = cell(inp, hidden[i], self.static_m)
            inp = torch.relu(hidden[i])
        return hidden[-1]

    def forward(self, batch):
        """batch: x [B, 4, N, 2], y [B, 4, N, 2] (scaled), slots [B, 8], dow/workday [B],
        ref_type/ref_week [B, 8, N, 2]. Returns scaled predictions [B, 4, N, 2]."""
        x = batch["x"]
        seq_len = x.shape[1]
        hidden = [None] * len(self.encoder)
        for t in range(seq_len):
            self._step(self.encoder, x[:, t], hidden, self._context(batch, t))
        dec_in = torch.zeros_like(x[:, 0])
        preds = []
        for t in range(self.horizon):
            out = self.output_layer(self._step(self.decoder, dec_in, hidden, self._context(batch, seq_len + t)))
            preds.append(out)
            teacher = (self.training and self.use_curriculum_learning and
                       random.random() < self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_step / self.cl_decay_steps)))
            dec_in = batch["y"][:, t] if teacher else out.detach()
        if self.training:
            self.global_step += 1
        return torch.stack(preds, 1)
