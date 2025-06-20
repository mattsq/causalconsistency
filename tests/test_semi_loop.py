import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from causal_consistency_nn.model import semi_loop
from causal_consistency_nn.model.semi_loop import EMConfig, train_em


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_z = nn.Linear(2, 1)
        self.fc_y = nn.Linear(2, 2)
        self.fc_x = nn.Linear(2, 1)

    def head_z_given_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float().unsqueeze(-1)
        return self.fc_z(torch.cat([x, y], dim=1))

    def head_y_given_xz(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.fc_y(torch.cat([x, z], dim=1))

    def head_x_given_yz(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = y.float().unsqueeze(-1)
        return self.fc_x(torch.cat([y, z], dim=1))


def create_data(n: int = 20) -> tuple[DataLoader, DataLoader]:
    x = torch.randn(n, 1)
    y = (x.squeeze() > 0).long()
    z = x + y.float().unsqueeze(-1)

    sup_ds = TensorDataset(x, y, z)
    unsup_ds = TensorDataset(x, z)
    sup_loader = DataLoader(sup_ds, batch_size=5)
    unsup_loader = DataLoader(unsup_ds, batch_size=5)
    return sup_loader, unsup_loader


def test_train_em_reduces_loss() -> None:
    sup_loader, unsup_loader = create_data()
    model = DummyModel()
    config = EMConfig(epochs=3, pretrain_epochs=1, lr=0.01)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    def eval_loss() -> float:
        total = 0.0
        for batch in sup_loader:
            x, y, z = batch
            z_pred = model.head_z_given_xy(x, y)
            y_logits = model.head_y_given_xz(x, z)
            x_pred = model.head_x_given_yz(y, z)
            loss = mse(z_pred, z) + ce(y_logits, y) + mse(x_pred, x)
            total += loss.item()
        return total

    loss_before = eval_loss()
    train_em(model, sup_loader, unsup_loader, config)
    loss_after = eval_loss()
    assert loss_after < loss_before


def test_soft_em_loss_decreases(capsys) -> None:
    sup_loader, unsup_loader = create_data()
    model = DummyModel()
    config = EMConfig(epochs=3, lr=0.01, beta=1.0, tau=0.5)

    train_em(model, sup_loader, unsup_loader, config)
    out_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "unsupervised_loss=" in line
    ]
    losses = [float(line.split("unsupervised_loss=")[1]) for line in out_lines]
    assert losses[-1] < losses[0]


class EntropyModel(nn.Module):
    def head_y_given_xz(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(x), 2)

    def head_z_given_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def head_x_given_yz(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(y), 1)


def test_entropy_term() -> None:
    model = EntropyModel()
    x = torch.randn(4, 1)
    z = torch.randn(4, 1)
    cfg = EMConfig(lambda1=0.0, lambda3=0.0, beta=1.0, tau=2.0)
    loss = semi_loop._unsupervised_step(model, (x, z), cfg, nn.MSELoss())
    entropy = -(torch.tensor([0.5, 0.5]) * torch.log(torch.tensor([0.5, 0.5]))).sum()
    expected = -cfg.tau * entropy
    assert torch.isclose(loss, expected, atol=1e-4)


def test_train_em_accepts_device() -> None:
    sup_loader, unsup_loader = create_data()
    model = DummyModel()
    train_em(model, sup_loader, unsup_loader, EMConfig(epochs=1), device="cpu")
    assert all(p.device.type == "cpu" for p in model.parameters())
