import torch

from networks.dgfa import (
    SlicedWassersteinPrior,
    masked_binary_cross_entropy,
    refine_pseudo_labels,
)


def test_running_barycenter_is_true_incremental_mean():
    prior = SlicedWassersteinPrior(2, num_projections=4, num_quantiles=8)
    first = torch.ones(1, 2, 2, 2)
    second = torch.full((1, 2, 2, 2), 3.0)
    first_q = prior._sorted_quantiles(prior._points(first))
    second_q = prior._sorted_quantiles(prior._points(second))
    prior.update(first)
    prior.update(second)
    torch.testing.assert_close(prior.barycenter, (first_q + second_q) / 2)
    assert prior.point_count.item() == 8


def test_dgfa_loss_and_pseudo_labels_are_finite():
    prior = SlicedWassersteinPrior(4, num_projections=8, num_quantiles=16)
    features = torch.randn(1, 4, 4, 4)
    prior.update(features)
    total, align, bn = prior.loss(features)
    assert torch.isfinite(torch.stack([total, align, bn])).all()
    logits = torch.randn(1, 2, 16, 16, requires_grad=True)
    pseudo, reliability = refine_pseudo_labels(logits, features)
    assert pseudo.shape == logits.shape
    assert reliability.shape == logits.shape
    assert torch.all(pseudo[:, 1] <= pseudo[:, 0])
    loss = masked_binary_cross_entropy(logits, pseudo, reliability)
    assert torch.isfinite(loss)
    loss.backward()
