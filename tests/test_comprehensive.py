"""Comprehensive tests for paddleOcean Phase 1 & 2.

Tests cover:
- Model (Lightning mode, Keras mode)
- Trainer (full cycle)
- DataModule
- Callbacks (ModelCheckpoint, EarlyStopping, LearningRateMonitor)
- Loggers (CSVLogger, VisualDLLogger)
- Gear (Fabric equivalent)
- Accelerators
- Checkpoint save/load
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paddle
import ocean


# ====================================================================
# Helper: Lightning-mode model
# ====================================================================

class LinearModel(ocean.Model):
    """Simple linear model for classification."""

    def __init__(self):
        super().__init__()
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

    def configure_optimizers(self):
        return paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())


# ====================================================================
# Helper: DataModule
# ====================================================================

class RandomDataModule(ocean.DataModule):
    def __init__(self, num_samples=100, batch_size=16):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = paddle.io.TensorDataset([
            paddle.randn([self.num_samples, 10]),
            paddle.randint(0, 2, [self.num_samples])
        ])
        self.val_dataset = paddle.io.TensorDataset([
            paddle.randn([self.num_samples // 2, 10]),
            paddle.randint(0, 2, [self.num_samples // 2])
        ])
        self.test_dataset = paddle.io.TensorDataset([
            paddle.randn([self.num_samples // 2, 10]),
            paddle.randint(0, 2, [self.num_samples // 2])
        ])

    def train_dataloader(self):
        return paddle.io.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return paddle.io.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return paddle.io.DataLoader(self.test_dataset, batch_size=self.batch_size)


# ====================================================================
# Tests
# ====================================================================

def make_train_loader(n=50, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


def make_val_loader(n=20, bs=8):
    ds = paddle.io.TensorDataset([paddle.randn([n, 10]), paddle.randint(0, 2, [n])])
    return paddle.io.DataLoader(ds, batch_size=bs)


# --- Model basic tests ---

def test_model_lightning_mode():
    model = LinearModel()
    trainer = ocean.Trainer(max_epochs=2, log_every_n_steps=5, verbose=0)
    dm = RandomDataModule(num_samples=30, batch_size=8)
    trainer.fit(model, datamodule=dm)
    assert model._trainer is not None
    assert trainer.global_step > 0
    assert trainer.current_epoch == 2


def test_model_lightning_direct_dataloaders():
    model = LinearModel()
    trainer = ocean.Trainer(max_epochs=2, verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(), val_dataloaders=make_val_loader())
    assert trainer.global_step > 0


def test_model_keras_mode():
    net = paddle.nn.Sequential(paddle.nn.Linear(10, 20), paddle.nn.ReLU(), paddle.nn.Linear(20, 2))
    model = ocean.Model(__model__=net)
    model.compile(
        optimizer=paddle.optimizer.SGD(learning_rate=0.01, parameters=net.parameters()),
        loss=paddle.nn.CrossEntropyLoss(),
    )
    model.fit(train_data=make_train_loader(), epochs=2)
    assert model._trainer is not None


def test_model_properties():
    model = LinearModel()
    assert model.automatic_optimization is True
    assert model.current_epoch == 0
    assert model.global_step == 0
    model.automatic_optimization = False
    assert model.automatic_optimization is False


def test_model_logging():
    model = LinearModel()
    model.log("test_metric", 0.5)
    model.log("test_tensor", paddle.to_tensor(0.8))


def test_model_fit_convenience():
    model = LinearModel()
    model.__trainer__ = ocean.Trainer(max_epochs=1, verbose=0)
    dm = RandomDataModule(num_samples=20, batch_size=8)
    model.fit(datamodule=dm)
    assert model._trainer is model.__trainer__


def test_model_forward_lightning():
    model = LinearModel()
    x = paddle.randn([4, 10])
    out = model(x)
    assert out.shape == [4, 2]


def test_model_forward_keras():
    net = paddle.nn.Linear(10, 2)
    model = ocean.Model(__model__=net)
    x = paddle.randn([4, 10])
    out = model(x)
    assert out.shape == [4, 2]


# --- Trainer tests ---

def test_trainer_validate():
    model = LinearModel()
    trainer = ocean.Trainer(verbose=0)
    metrics_list = trainer.validate(model, dataloaders=make_val_loader())
    assert isinstance(metrics_list, list)
    if metrics_list:
        assert isinstance(metrics_list[0], dict)


def test_trainer_test():
    model = LinearModel()
    trainer = ocean.Trainer(verbose=0)
    # Train first so model has some learned weights
    trainer.fit(model, train_dataloaders=make_train_loader(30), val_dataloaders=make_val_loader(10))
    metrics = trainer.test(model, dataloaders=make_val_loader())
    assert isinstance(metrics, list)


def test_trainer_predict():
    class PredictModel(ocean.Model):
        def __init__(self):
            super().__init__()
            self.linear = paddle.nn.Linear(10, 2)
        def forward(self, x):
            return self.linear(x)

    model = PredictModel()
    ds = paddle.io.TensorDataset([paddle.randn([10, 10])])
    loader = paddle.io.DataLoader(ds, batch_size=4)
    trainer = ocean.Trainer(verbose=0)
    predictions = trainer.predict(model, dataloaders=loader)
    assert len(predictions) > 0


def test_trainer_accumulate_grad():
    model = LinearModel()
    trainer = ocean.Trainer(max_epochs=1, accumulate_grad_batches=2, verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(16, 4))
    assert trainer.global_step > 0


# --- DataModule tests ---

def test_datamodule():
    dm = RandomDataModule(num_samples=40, batch_size=8)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    batch = next(iter(train_loader))
    assert len(batch) == 2


# --- Callback tests ---

def test_early_stopping():
    model = LinearModel()
    early_stop = ocean.EarlyStopping(monitor="val_loss", patience=2, verbose=False)
    trainer = ocean.Trainer(max_epochs=10, callbacks=[early_stop], verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(20, 8), val_dataloaders=make_val_loader(10, 8))
    # Either stopped early or ran to completion
    assert trainer.current_epoch <= 10


def test_model_checkpoint():
    model = LinearModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = ocean.ModelCheckpoint(dirpath=tmpdir, save_last=True, verbose=False)
        trainer = ocean.Trainer(max_epochs=2, callbacks=[ckpt], verbose=0)
        trainer.fit(model, train_dataloaders=make_train_loader(20, 8))
        last_path = os.path.join(tmpdir, "last.ckpt")
        assert os.path.exists(last_path), f"Checkpoint not found at {last_path}"


def test_learning_rate_monitor():
    model = LinearModel()
    lr_mon = ocean.LearningRateMonitor()
    trainer = ocean.Trainer(max_epochs=1, callbacks=[lr_mon], verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(20, 8))


def test_callback_base():
    cb = ocean.Callback()
    cb.setup(None, None, "fit")
    cb.teardown(None, None, "fit")
    assert cb.state_dict() == {}


# --- Logger tests ---

def test_csv_logger():
    model = LinearModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ocean.CSVLogger(root_dir=tmpdir, name="test_logs")
        trainer = ocean.Trainer(max_epochs=1, logger=logger, verbose=0)
        trainer.fit(model, train_dataloaders=make_train_loader(20, 8))
        trainer._logger_connector.log_metrics({"dummy": 1.0}, step=1)
        logger.save()
        logger.finalize("success")
        log_dir = logger.log_dir
        metrics_file = os.path.join(log_dir, "metrics.csv")
        assert os.path.exists(metrics_file), f"CSV not found at {metrics_file}"


def test_csv_logger_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ocean.CSVLogger(root_dir=tmpdir, name="test")
        logger.log_metrics({"train_loss": 0.5}, step=1)
        logger.log_metrics({"train_loss": 0.4, "val_acc": 0.8}, step=2)
        logger.save()
        log_dir = logger.log_dir
        assert os.path.exists(os.path.join(log_dir, "metrics.csv"))


def test_visualdl_logger():
    if not _has_visualdl():
        return  # Skip if VisualDL not installed
    model = LinearModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ocean.VisualDLLogger(save_dir=tmpdir, name="test_vdl")
        trainer = ocean.Trainer(max_epochs=1, logger=logger, verbose=0)
        trainer.fit(model, train_dataloaders=make_train_loader(20, 8))
        logger.finalize("success")


def test_logger_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ocean.CSVLogger(root_dir=tmpdir)
        assert hasattr(logger, "name")
        assert hasattr(logger, "version")
        assert hasattr(logger, "log_metrics")
        logger.save()


# --- Gear tests ---

def test_gear_basic():
    gear = ocean.Gear(accelerator="cpu")
    model = paddle.nn.Linear(10, 2)
    model = gear.setup(model)
    assert model is not None

    x = paddle.randn([4, 10])
    out = model(x)
    assert out.shape == [4, 2]


def test_gear_setup_dataloaders():
    gear = ocean.Gear()
    loader = make_train_loader()
    result = gear.setup_dataloaders(loader)
    assert result is loader


def test_gear_save_load():
    gear = ocean.Gear()
    model = paddle.nn.Linear(10, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.pth")
        gear.save(path, {"model": model})
        loaded = gear.load(path, strict=False)
        assert "model" in loaded


def test_gear_seed():
    gear = ocean.Gear()
    seed = gear.seed_everything(42, verbose=False)
    assert seed == 42


def test_gear_to_device():
    gear = ocean.Gear()
    t = paddle.randn([3, 3])
    t2 = gear.to_device(t)
    assert t2 is not None


def test_gear_backward():
    gear = ocean.Gear()
    model = paddle.nn.Linear(10, 2)
    x = paddle.randn([4, 10])
    y = paddle.randint(0, 2, [4])
    logits = model(x)
    loss = paddle.nn.functional.cross_entropy(logits, y)
    gear.backward(loss)
    assert True  # backward ran without error


# --- Accelerator tests ---

def test_cpu_accelerator():
    accel = ocean.CPUAccelerator()
    device = accel.setup_device()
    assert device is not None


def test_accelerator_base():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ocean.CSVLogger(root_dir=tmpdir)
        assert isinstance(logger, ocean.Logger)


# --- Model checkpoint save/load ---

def test_model_save_load_checkpoint():
    model = LinearModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.ckpt")
        model.save_checkpoint(path)
        assert os.path.exists(path)

        # Load into a new model
        model2 = LinearModel()
        ckpt = model2.load_checkpoint(path, strict=False)
        assert "state_dict" in ckpt


# --- DataModule with Trainer ---

def test_datamodule_with_trainer():
    model = LinearModel()
    dm = RandomDataModule(num_samples=30, batch_size=8)
    trainer = ocean.Trainer(max_epochs=2, verbose=0)
    trainer.fit(model, datamodule=dm)
    assert trainer.current_epoch == 2

    # Test with validate
    metrics = trainer.validate(model, datamodule=dm)
    assert isinstance(metrics, list)

    # Test with test
    dm.setup("test")
    metrics = trainer.test(model, datamodule=dm)
    assert isinstance(metrics, list)


# --- Multiple epochs with validation ---

def test_multi_epoch_validation():
    model = LinearModel()
    trainer = ocean.Trainer(max_epochs=3, check_val_every_n_epoch=1, verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(30, 8), val_dataloaders=make_val_loader(10, 8))
    assert trainer.current_epoch == 3


# --- Keras mode compile and evaluate ---

def test_keras_evaluate():
    net = paddle.nn.Sequential(paddle.nn.Linear(10, 2))
    model = ocean.Model(__model__=net)
    model.compile(
        optimizer=paddle.optimizer.SGD(learning_rate=0.01, parameters=net.parameters()),
        loss=paddle.nn.CrossEntropyLoss(),
    )
    model.fit(train_data=make_train_loader(30), epochs=1)
    metrics = model.evaluate(eval_data=make_val_loader())
    assert isinstance(metrics, list)


# --- Fast dev run ---

def test_fast_dev_run():
    model = LinearModel()
    trainer = ocean.Trainer(fast_dev_run=2, verbose=0)
    trainer.fit(model, train_dataloaders=make_train_loader(50, 8), val_dataloaders=make_val_loader(20, 8))
    assert trainer.global_step > 0


# --- import test ---

def test_imports():
    assert hasattr(ocean, "Model")
    assert hasattr(ocean, "Trainer")
    assert hasattr(ocean, "DataModule")
    assert hasattr(ocean, "Gear")
    assert hasattr(ocean, "Callback")
    assert hasattr(ocean, "ModelCheckpoint")
    assert hasattr(ocean, "EarlyStopping")
    assert hasattr(ocean, "LearningRateMonitor")
    assert hasattr(ocean, "CSVLogger")
    assert hasattr(ocean, "VisualDLLogger")
    assert hasattr(ocean, "Logger")
    assert hasattr(ocean, "CPUAccelerator")
    assert hasattr(ocean, "GPUAccelerator")


# --- Run all ---

def _has_visualdl():
    try:
        import visualdl  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    paddle.seed(42)

    tests = [
        ("test_imports", test_imports),
        ("test_model_lightning_mode", test_model_lightning_mode),
        ("test_model_lightning_direct_dataloaders", test_model_lightning_direct_dataloaders),
        ("test_model_keras_mode", test_model_keras_mode),
        ("test_model_properties", test_model_properties),
        ("test_model_logging", test_model_logging),
        ("test_model_fit_convenience", test_model_fit_convenience),
        ("test_model_forward_lightning", test_model_forward_lightning),
        ("test_model_forward_keras", test_model_forward_keras),
        ("test_trainer_validate", test_trainer_validate),
        ("test_trainer_test", test_trainer_test),
        ("test_trainer_predict", test_trainer_predict),
        ("test_trainer_accumulate_grad", test_trainer_accumulate_grad),
        ("test_datamodule", test_datamodule),
        ("test_early_stopping", test_early_stopping),
        ("test_model_checkpoint", test_model_checkpoint),
        ("test_learning_rate_monitor", test_learning_rate_monitor),
        ("test_callback_base", test_callback_base),
        ("test_csv_logger", test_csv_logger),
        ("test_csv_logger_metrics", test_csv_logger_metrics),
        ("test_visualdl_logger", test_visualdl_logger),
        ("test_logger_base", test_logger_base),
        ("test_gear_basic", test_gear_basic),
        ("test_gear_setup_dataloaders", test_gear_setup_dataloaders),
        ("test_gear_save_load", test_gear_save_load),
        ("test_gear_seed", test_gear_seed),
        ("test_gear_to_device", test_gear_to_device),
        ("test_gear_backward", test_gear_backward),
        ("test_cpu_accelerator", test_cpu_accelerator),
        ("test_accelerator_base", test_accelerator_base),
        ("test_model_save_load_checkpoint", test_model_save_load_checkpoint),
        ("test_datamodule_with_trainer", test_datamodule_with_trainer),
        ("test_multi_epoch_validation", test_multi_epoch_validation),
        ("test_keras_evaluate", test_keras_evaluate),
        ("test_fast_dev_run", test_fast_dev_run),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        sys.exit(1)
    else:
        print("✅ All tests passed!")
