"""Forward-pass equivalence of metro.model.Net (plain PVCGN mode) with the authors' reference Net."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference_PVCGN"))

from ggnn.multigraph import Net as RefNet  # noqa: E402
from ggnn_utils import collate_wrapper  # noqa: E402
from metro.data import load_graphs  # noqa: E402
from metro.model import Net  # noqa: E402


def reference_edges(adjs):
    adj = np.stack(adjs, -1).astype(np.float64)
    for r in range(adj.shape[-1]):
        np.fill_diagonal(adj[..., r], 0)
    adj = adj / (adj.sum(axis=0) + 1e-18)
    src, dst = adj.sum(-1).nonzero()
    return torch.tensor(np.array([src, dst])), torch.tensor(adj[adj.sum(-1) != 0], dtype=torch.float)


def copy_cell(ref, mine):
    for conv_r, conv_m in ((ref.cheb_i.rgcn_layers[0], mine.conv_i), (ref.cheb_h.rgcn_layers[0], mine.conv_h)):
        conv_m.basis.data.copy_(conv_r.basis.data)
        conv_m.att.data.copy_(conv_r.att.data)
        conv_m.root.data.copy_(conv_r.root.data)
    for name in ("bias_r", "bias_i", "bias_n", "bias_r_g", "bias_i_g", "bias_n_g"):
        getattr(mine, name).data.copy_(getattr(ref, name).data)
    for name in ("mlpi", "mlph", "global_i", "global_h", "mlpatt", "ln"):
        getattr(mine, name).load_state_dict(getattr(ref, name).state_dict())


def test_forward_matches_reference():
    torch.manual_seed(0)
    g = load_graphs("hz")
    adjs = [g["conn"], g["sml"], g["cor"]]
    cfg = {"model": dict(num_nodes=80, output_dim=2, rnn_units=32, input_dim=2, num_rnn_layers=2, seq_len=4,
                         horizon=4, use_curriculum_learning=False, cl_decay_steps=200, K=1, num_bases=3,
                         num_relations=3, global_fusion=True, dropout_type="zoneout", dropout_prob=0.0)}
    ref = RefNet(cfg).eval()
    mine = Net(80, np.stack(adjs), units=32, use_curriculum_learning=False).eval()
    for r, m in zip(list(ref.encoder_cells) + list(ref.decoder_cells), list(mine.encoder) + list(mine.decoder)):
        copy_cell(r, m)
    mine.output_layer.load_state_dict(ref.output_layer.state_dict())

    x = np.random.default_rng(0).normal(size=(3, 4, 80, 2)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=(3, 4, 80, 2)).astype(np.float32)
    edge_index, edge_attr = reference_edges(adjs)
    seqs, _ = collate_wrapper(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, device="cpu")
    with torch.no_grad():
        out_ref = ref(seqs)
        out_mine = mine({"x": torch.tensor(x), "y": torch.tensor(y), "slots": torch.zeros(3, 8, dtype=torch.long)})
    assert out_ref.shape == out_mine.shape
    assert torch.allclose(out_ref, out_mine, atol=1e-5), (out_ref - out_mine).abs().max()


if __name__ == "__main__":
    test_forward_matches_reference()
    print("equivalent")
