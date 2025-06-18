import torch
from torch import nn

from causal_consistency_nn.metrics import average_treatment_effect, log_likelihood
from causal_consistency_nn.serve import predict_z, counterfactual_z, impute_y


def _create_synth(n: int = 50, noise: float = 0.1):
    x = torch.randn(n, 1)
    y = (x.squeeze() > 0).long()
    z = x + y.float().unsqueeze(-1) + torch.randn_like(x) * noise
    return x, y, z


class OracleModel(nn.Module):
    def __init__(self, noise: float = 0.1) -> None:
        super().__init__()
        self.noise = noise

    def head_z_given_xy(self, x: torch.Tensor, y: torch.Tensor):
        mu = x + y.float().unsqueeze(-1)
        sigma = torch.full_like(mu, self.noise)
        return torch.distributions.Normal(mu, sigma)

    def head_y_given_xz(self, x: torch.Tensor, z: torch.Tensor):
        logits = (
            torch.stack([-(z - x).squeeze(), (z - x).squeeze()], dim=-1) / self.noise
        )
        return torch.distributions.Categorical(logits=logits)


class TestMetrics:
    def test_ate_and_ll(self) -> None:
        x, y, z = _create_synth()
        model = OracleModel()
        ate_val = average_treatment_effect(model, x)
        assert torch.isclose(ate_val, torch.tensor(1.0), atol=0.1)
        ll = log_likelihood(model.head_z_given_xy(x, y), z)
        assert ll > -1.0

    def test_serving_helpers(self) -> None:
        x, y, z = _create_synth(n=10)
        model = OracleModel()
        preds = predict_z(model, x, y)
        assert preds.shape == z.shape
        cf = counterfactual_z(model, x, 1 - y)
        assert cf.shape == z.shape
        probs = impute_y(model, x, z)
        assert probs.shape == (x.size(0), 2)
