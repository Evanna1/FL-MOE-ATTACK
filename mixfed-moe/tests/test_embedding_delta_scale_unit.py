import torch
from torch import nn

from mixfedmoe_fl.client import amplify_embedding_delta, snapshot_input_embeddings


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 3)

    def get_input_embeddings(self):
        return self.emb


def test_amplify_delta_cancels_fedavg_dilution():
    n = 8
    model = _M()
    e_g = model.emb.weight.detach().clone()
    snap = snapshot_input_embeddings(model)
    target = e_g.clone()
    target[2] += 0.5
    with torch.no_grad():
        model.emb.weight.copy_(target)
    amplify_embedding_delta(snap, n)
    uploaded = model.emb.weight.detach()
    assert torch.allclose(uploaded, e_g + n * (target - e_g), atol=1e-6)
    aggregated = uploaded / n + (n - 1) / n * e_g
    assert torch.allclose(aggregated, target, atol=1e-5)


def test_amplify_delta_only_selected_token_rows():
    model = _M()
    e_g = model.emb.weight.detach().clone()
    snap = snapshot_input_embeddings(model)
    trained = e_g.clone()
    trained[1] += 0.5
    trained[3] += 0.25
    with torch.no_grad():
        model.emb.weight.copy_(trained)
    amplify_embedding_delta(snap, 4, token_ids=[1])
    out = model.emb.weight.detach()
    assert torch.allclose(out[1], e_g[1] + 4 * (trained[1] - e_g[1]), atol=1e-6)
    assert torch.allclose(out[3], trained[3])  # non-trigger row untouched
    assert torch.allclose(out[0], e_g[0])
