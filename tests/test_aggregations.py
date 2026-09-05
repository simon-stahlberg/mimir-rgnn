import torch

from pymimir_rgnn import SmoothMaximumAggregation


def test_smooth_maximum_preserves_gradients_and_empty_nodes() -> None:
    node_embeddings = torch.zeros((4, 2))
    messages = torch.tensor(
        [[1.0, -1.0], [3.0, 2.0], [2.0, 0.0]],
        requires_grad=True,
    )
    indices = torch.tensor([2, 0, 2])

    output = SmoothMaximumAggregation()(node_embeddings, messages, indices)
    output.sum().backward()

    expected_empty = torch.log(torch.tensor(1E-16)) / 12.0
    assert output.shape == node_embeddings.shape
    assert torch.allclose(output[1], expected_empty.expand(2))
    assert torch.allclose(output[3], expected_empty.expand(2))
    assert messages.grad is not None
    assert torch.isfinite(messages.grad).all()

