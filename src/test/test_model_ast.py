from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

import src.model.pretrained.ast as ast_module
from src.model.pretrained.ast import ASTEmbedder


@pytest.mark.unit
def test_fix_tdim_pads_short_ast_features() -> None:
    feats = torch.arange(2 * 1001 * 128, dtype=torch.float32).reshape(2, 1001, 128)

    fixed = ASTEmbedder._fix_tdim(feats, 1024)

    assert fixed.shape == (2, 1024, 128)
    assert torch.equal(fixed[:, :1001, :], feats)
    assert torch.count_nonzero(fixed[:, 1001:, :]) == 0


@pytest.mark.unit
def test_fix_tdim_center_crops_long_ast_features() -> None:
    feats = torch.arange(1 * 1030 * 4, dtype=torch.float32).reshape(1, 1030, 4)

    fixed = ASTEmbedder._fix_tdim(feats, 1024)

    assert fixed.shape == (1, 1024, 4)
    assert torch.equal(fixed, feats[:, 3:1027, :])


@pytest.mark.unit
def test_fix_tdim_keeps_matching_ast_features() -> None:
    feats = torch.randn(3, 1024, 128)

    fixed = ASTEmbedder._fix_tdim(feats, 1024)

    assert fixed.shape == feats.shape
    assert torch.equal(fixed, feats)


class _DummyAstModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_length=1024, hidden_size=768)

    def forward(self, input_values: torch.Tensor):
        pooled = input_values.mean(dim=(1, 2)).unsqueeze(1).repeat(1, 768)
        return SimpleNamespace(pooler_output=pooled)


@pytest.mark.unit
def test_ast_forward_output_can_flow_into_trainable_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ast_module.ASTModel,
        "from_pretrained",
        lambda *args, **kwargs: _DummyAstModel(),
    )

    embedder = ASTEmbedder(device="cpu")
    wave = torch.zeros(2, 16000 * 10, dtype=torch.float32)
    head = nn.Linear(768, 2)

    emb = embedder(wave)
    loss = head(emb).sum()
    loss.backward()

    assert head.weight.grad is not None

