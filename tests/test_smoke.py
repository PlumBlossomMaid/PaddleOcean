"""Smoke tests for paddleOcean Phase 1 core abstractions.

Tests both Keras mode (ocean.Model with __model__) and Lightning mode
(user subclass of ocean.Model with hooks).
"""

import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle

import ocean

# ====================================================================
# Lightning-mode model
# ====================================================================


class LinearModel(ocean.Model):
    """A simple linear model in Lightning mode."""

    def __init__(self):
        super().__init__()  # __model__ = None
        self.linear = paddle.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = paddle.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(axis=1) == y).astype(paddle.float32).mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


# ====================================================================
# Keras-mode model
# ====================================================================


def make_keras_model():
    net = paddle.nn.Sequential(
        paddle.nn.Linear(10, 20),
        paddle.nn.ReLU(),
        paddle.nn.Linear(20, 2),
    )
    model = ocean.Model(model=net)
    model.prepare(
        optimizer=paddle.optimizer.SGD(learning_rate=0.01, parameters=net.parameters()),
        loss=paddle.nn.CrossEntropyLoss(),
    )
    return model


# ====================================================================
# DataModule
# ====================================================================


class RandomDataModule(ocean.DataModule):
    """Simple DataModule that generates random data."""

    def __init__(self, num_samples=100, batch_size=16):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage):
        # Synthetic dataset
        self.train_dataset = paddle.io.TensorDataset([
            paddle.randn([self.num_samples, 10]),
            paddle.randint(0, 2, [self.num_samples]),
        ])
        self.val_dataset = paddle.io.TensorDataset([
            paddle.randn([self.num_samples // 2, 10]),
            paddle.randint(0, 2, [self.num_samples // 2]),
        ])

    def train_dataloader(self):
        return paddle.io.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return paddle.io.DataLoader(self.val_dataset, batch_size=self.batch_size)


# ====================================================================
# Tests
# ====================================================================


def test_model_lightning_mode():
    """Test Lightning mode: subclass ocean.Model with hooks."""
    model = LinearModel()
    trainer = ocean.Trainer(max_epochs=2, log_every_n_steps=5, verbose=1)
    dm = RandomDataModule(num_samples=50, batch_size=8)
    trainer.fit(model, datamodule=dm)

    assert model._trainer is not None
    assert trainer.dataloader_step > 0
    assert trainer.current_epoch == 2


def test_model_lightning_mode_direct_dataloaders():
    """Test Lightning mode with direct dataloader arguments."""
    model = LinearModel()

    train_dataset = paddle.io.TensorDataset([paddle.randn([50, 10]), paddle.randint(0, 2, [50])])
    val_dataset = paddle.io.TensorDataset([paddle.randn([20, 10]), paddle.randint(0, 2, [20])])
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=8)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=8)

    trainer = ocean.Trainer(max_epochs=2, log_every_n_steps=10, verbose=0)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert trainer.dataloader_step > 0


def test_model_keras_mode():
    """Test Keras mode: ocean.Model with __model__ and compile()."""
    model = make_keras_model()
    assert model.__model__ is not None

    train_dataset = paddle.io.TensorDataset([paddle.randn([50, 10]), paddle.randint(0, 2, [50])])
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=8)

    model.fit(train_data=train_loader, epochs=2)

    assert model._trainer is not None
    assert model.__trainer__ is not None


def test_trainer_validate():
    """Test Trainer.validate() standalone."""
    model = LinearModel()
    dm = RandomDataModule(num_samples=30, batch_size=8)
    dm.setup("validate")
    trainer = ocean.Trainer(verbose=0)
    metrics = trainer.validate(model, dataloaders=dm.val_dataloader())
    assert isinstance(metrics, list)
    if metrics:
        assert isinstance(metrics[0], dict)


def test_trainer_predict():
    """Test Trainer.predict() standalone."""

    class PredictModel(ocean.Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    model = PredictModel()
    dataset = paddle.io.TensorDataset([paddle.randn([10, 10])])
    loader = paddle.io.DataLoader(dataset, batch_size=4)

    trainer = ocean.Trainer(verbose=0)
    predictions = trainer.predict(model, dataloaders=loader)
    assert len(predictions) > 0


def test_datamodule_basic():
    """Test DataModule setup and dataloaders."""
    dm = RandomDataModule(num_samples=40, batch_size=8)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))
    assert len(batch) == 2  # x, y
    assert batch[0].shape[0] == 8


def test_model_logging():
    """Test that model.log() doesn't crash."""
    model = LinearModel()

    # Log without trainer attached (should be no-op)
    model.log("test_metric", 0.5)
    model.log("test_tensor", paddle.to_tensor(0.8))

    # Log with trainer attached
    trainer = ocean.Trainer(verbose=0, max_epochs=1)
    dm = RandomDataModule(num_samples=20, batch_size=8)
    trainer.fit(model, datamodule=dm)


def test_model_fit_convenience():
    """Test Model.fit() convenience method (uses __trainer__)."""
    model = LinearModel()
    model.__trainer__ = ocean.Trainer(max_epochs=1, verbose=0)

    dm = RandomDataModule(num_samples=20, batch_size=8)
    model.fit(datamodule=dm)

    assert model._trainer is model.__trainer__


def test_import_ocean():
    """Test that 'import ocean' works."""
    assert hasattr(ocean, "Model")
    assert hasattr(ocean, "Trainer")
    assert hasattr(ocean, "DataModule")


if __name__ == "__main__":
    paddle.seed(42)

    test_import_ocean()
    print("✓ test_import_ocean")

    test_datamodule_basic()
    print("✓ test_datamodule_basic")

    test_model_logging()
    print("✓ test_model_logging")

    test_model_lightning_mode()
    print("✓ test_model_lightning_mode")

    test_model_lightning_mode_direct_dataloaders()
    print("✓ test_model_lightning_mode_direct_dataloaders")

    test_model_keras_mode()
    print("✓ test_model_keras_mode")

    test_trainer_validate()
    print("✓ test_trainer_validate")

    test_trainer_predict()
    print("✓ test_trainer_predict")

    test_model_fit_convenience()
    print("✓ test_model_fit_convenience")

    print("\n✅ All smoke tests passed!")
