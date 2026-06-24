"""Smoke test: verify Ocean XPUAccelerator can train on XPU P800.

All tests are skipped when XPU is not available.
"""

import pytest

import ocean

pytestmark = pytest.mark.skipif(
    not ocean.accelerators.XPUAccelerator.is_available(),
    reason="XPU not available",
)


def test_xpu_available():
    """XPU P800 device detection."""
    assert ocean.accelerators.XPUAccelerator.is_available()
    assert ocean.accelerators.XPUAccelerator.auto_device_count() == 1


def test_xpu_setup_device():
    """XPU place creation."""
    acc = ocean.accelerators.XPUAccelerator()
    place = acc.setup_device()
    assert isinstance(place, ocean.XPUPlace)


def test_xpu_tensor_ops():
    """Basic tensor operations on XPU."""
    ocean.set_device("xpu")
    x = ocean.randn([100, 100])
    y = ocean.randn([100, 100])
    z = x + y
    assert "xpu" in str(z.place).lower()


def test_xpu_trainer():
    """End-to-end training on XPU with Ocean Trainer (Lightning mode)."""
    ocean.seed_everything(42)

    class XPUModel(ocean.Model):
        def __init__(self):
            super().__init__()
            self.net = ocean.nn.Sequential(
                ocean.nn.Linear(10, 64),
                ocean.nn.ReLU(),
                ocean.nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            out = self(x)
            loss = ocean.nn.functional.cross_entropy(out, y.squeeze())
            return loss

        def configure_optimizers(self):
            return ocean.optimizer.Adam(learning_rate=0.001, parameters=self.parameters())

    x = ocean.randn([32, 10])
    y = ocean.randint(0, 2, [32])

    train_dataset = ocean.io.TensorDataset([x, y])
    train_loader = ocean.io.DataLoader(train_dataset, batch_size=8)

    model = XPUModel()
    trainer = ocean.Trainer(
        accelerator="xpu",
        max_epochs=2,
        enable_progress_bar=False,
        log_every_n_steps=999,
    )
    trainer.fit(model, train_loader)


def test_xpu_gear():
    """Manual training loop on XPU with Gear."""
    ocean.seed_everything(42)

    net = ocean.nn.Sequential(
        ocean.nn.Linear(10, 64),
        ocean.nn.ReLU(),
        ocean.nn.Linear(64, 2),
    )
    gear = ocean.Gear(accelerator="xpu")
    net = gear.setup(net)

    x = ocean.randn([16, 10])
    y = ocean.randint(0, 2, [16])

    x, y = gear.to_device([x, y])

    for _ in range(3):
        out = net(x)
        loss = ocean.nn.functional.cross_entropy(out, y)
        gear.backward(loss)
