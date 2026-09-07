import torch

from networks.semantic_anchor import SemanticAnchorInducer


def test_semantic_anchor_shape_parameter_budget_and_gradient():
    module = SemanticAnchorInducer(torch.randn(5, 512), visual_dim=256)
    features = torch.randn(2, 256, 16, 16)
    anchor, loss = module(features, (512, 512))
    assert anchor.shape == (2, 3, 512, 512)
    assert anchor.min() >= -1 and anchor.max() <= 1
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    assert trainable < 100_000
    (anchor.mean() + loss).backward()
    assert any(p.grad is not None for p in module.parameters())
