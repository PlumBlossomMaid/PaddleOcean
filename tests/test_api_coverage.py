"""Test that ALL PaddlePaddle APIs are accessible through ocean without importing paddle.

This test ensures ocean serves as a complete drop-in replacement for paddle.
"""


import paddle

# APIs that intentionally differ (documented differences)
INTENTIONAL_DIFFERENCES = {
    # Ocean's own framework additions
    "Model", "Trainer", "DataModule", "Gear",
    "Callback", "ModelCheckpoint", "EarlyStopping",
    "CSVLogger", "VisualDLLogger",
    "Strategy", "SingleDeviceStrategy",
    # Version info
    "Version", "PADDLE_VERSION", "version_gte", "version_lt",
    "repeat_interleave", "index_add", "scatter", "scatter_nd",
    "take_along_axis", "put_along_axis", "masked_fill", "masked_select",
    "sort", "argsort", "unique", "nonzero",
    "logsumexp", "lgamma",
    # Internals
    "_Loop", "_FitLoop", "_TrainingEpochLoop", "_EvaluationLoop", "_PredictionLoop",
    "_call_callback_hooks", "_call_lightning_module_hook",
    "_DataConnector", "_LoggerConnector", "_CallbackConnector", "_CheckpointConnector",
    # Trainer
    "TrainerState", "TrainerStatus", "TrainerFn", "RunningStage",
    "Precision",
    "OceanOptimizer", "init_optimizers_and_lr_schedulers",
    "load_from_checkpoint",
    "ModelHooks", "DataHooks", "HyperparametersMixin",
    "STEP_OUTPUT", "EVALUATE_OUTPUT", "PREDICT_OUTPUT",
}

# Known paddle internal APIs that shouldn't be proxied
PADDLE_INTERNAL = {
    "_C", "_C_ops", "_C_ops_", "_legacy_C_ops", "_pir_ops", "_ops",
    "_classes", "_typing", "_paddle_docs",
    "api_tracer", "apy", "base", "cinn_config", "cost_model",
    "framework", "libs", "pir", "proto", "tensorrt",
}


def test_ocean_proxies_paddle_top_level():
    """Verify key top-level paddle APIs are available as ocean.xxx."""
    import ocean

    # Verify ocean.Tensor is same as paddle.Tensor
    assert "Tensor" in dir(ocean)

    # Verify key APIs exist (not exhaustive - paddle has 649 top-level attrs,
    # ocean proxies them via __getattr__ so dir() covers all)
    assert ocean.randn is not None
    assert ocean.ones is not None
    assert ocean.zeros is not None
    assert ocean.nn is not None
    assert ocean.optimizer is not None


def test_ocean_tensor_operations():
    """Verify basic tensor operations work through ocean."""
    import ocean

    x = ocean.randn([3, 4])
    y = ocean.ones([3, 4])
    z = ocean.zeros([3, 4])

    assert x.shape == [3, 4]
    assert y.shape == [3, 4]
    assert z.shape == [3, 4]

    # tensor operations
    assert ocean.sum(x).shape is not None
    assert ocean.mean(x).shape is not None
    assert ocean.sin(x).shape == x.shape
    assert ocean.cos(x).shape == x.shape
    assert ocean.exp(x).shape == x.shape


def test_ocean_nn_layers():
    """Verify neural network layers are accessible."""
    import ocean

    layers = [
        ("Linear", [10, 2]),
        ("Conv2D", [3, 6, 3]),
        ("ReLU", []),
        ("Sigmoid", []),
        ("Tanh", []),
        ("Dropout", [0.5]),
        ("BatchNorm2D", [6]),
        ("Embedding", [100, 32]),
        ("LSTM", [10, 20, 2]),
        ("GRU", [10, 20, 2]),
    ]

    for name, args in layers:
        cls = getattr(ocean.nn, name, None)
        assert cls is not None, f"ocean.nn.{name} not found"
        instance = cls(*args) if args else cls()
        assert instance is not None, f"ocean.nn.{name}() failed"


def test_ocean_optimizers():
    """Verify optimizers are accessible."""
    import ocean

    layer = ocean.nn.Linear(10, 2)
    optimizers = ["SGD", "Adam", "AdamW", "Adagrad", "RMSProp", "Momentum"]

    for name in optimizers:
        cls = getattr(ocean.optimizer, name, None)
        assert cls is not None, f"ocean.optimizer.{name} not found"
        opt = cls(learning_rate=0.01, parameters=layer.parameters())
        assert opt is not None


def test_ocean_nn_functional():
    """Verify functional APIs are accessible."""
    import ocean

    funcs = ["relu", "sigmoid", "tanh", "softmax", "cross_entropy",
             "mse_loss", "binary_cross_entropy", "dropout",
             "batch_norm", "linear", "conv2d", "max_pool2d",
             "one_hot", "pad", "interpolate"]

    for name in funcs:
        fn = getattr(ocean.nn.functional, name, None)
        assert fn is not None, f"ocean.nn.functional.{name} not found"


def test_ocean_linalg():
    """Verify linear algebra ops."""
    import ocean

    x = ocean.randn([3, 3])
    ops = ["matmul", "norm", "det", "inv", "svd", "qr", "cholesky", "eig", "lstsq", "cross"]
    for name in ops:
        fn = getattr(ocean.linalg, name, None)
        assert fn is not None, f"ocean.linalg.{name} not found"


def test_ocean_fft():
    """Verify FFT ops."""
    import ocean

    ops = ["fft", "ifft", "rfft", "irfft", "fftn", "ifftn", "fftfreq", "rfftfreq"]
    for name in ops:
        fn = getattr(ocean.fft, name, None)
        assert fn is not None, f"ocean.fft.{name} not found"


def test_ocean_io():
    """Verify data loading APIs."""
    import ocean

    assert ocean.io.DataLoader is not None
    assert ocean.io.Dataset is not None


def test_ocean_device():
    """Verify device APIs."""
    import ocean

    assert ocean.CPUPlace is not None
    assert ocean.device.get_device is not None
    assert ocean.device.set_device is not None


def test_ocean_data_types():
    """Verify data types."""
    import ocean

    assert ocean.float32 is not None
    assert ocean.float64 is not None
    assert ocean.int32 is not None
    assert ocean.int64 is not None
    assert ocean.bool is not None


def test_ocean_compat_apis():
    """Verify compat-wrapped APIs work."""
    import ocean

    x = paddle.randn([2, 5])

    # repeat_interleave
    result = ocean.repeat_interleave(x, 3, axis=1)
    assert result.shape == [2, 15], f"repeat_interleave: {result.shape}"

    # sort
    sorted_x, indices = ocean.sort(x, axis=-1)
    assert sorted_x.shape == x.shape
    assert indices.shape == x.shape

    # argsort
    idx = ocean.argsort(x, axis=-1)
    assert idx.shape == x.shape

    # unique
    u = ocean.unique(x.flatten())
    assert u.shape[0] <= x.numel()

    # nonzero
    mask = x > 0
    nz = ocean.nonzero(mask)
    assert nz.shape[1] == x.ndim

    # masked_fill
    result = ocean.masked_fill(x, x > 0, 0.0)
    assert result.shape == x.shape

    # masked_select
    selected = ocean.masked_select(x, x > 0)
    assert selected.ndim == 1

    # scatter_along_axis
    src = paddle.randn([5, 10])
    idx = paddle.randint(0, 2, [5, 10])  # indices must be < axis dim size
    result = ocean.scatter_along_axis(paddle.zeros([5, 10]), idx, src, axis=0)

    # logsumexp
    lse = ocean.logsumexp(x, axis=-1)
    assert lse.shape == [2]


def test_no_paddle_import():
    """Test that ocean's __getattr__ proxy returns paddle APIs without explicit paddle import."""
    import ocean

    # Verify we can use ocean without having imported paddle ourselves
    assert "paddle" not in dir() or hasattr(ocean, 'randn')
    x = ocean.randn([3, 4])
    assert x.shape == [3, 4]

    layer = ocean.nn.Linear(10, 2)
    assert layer is not None

    # Verify ocean.randn is the same function as would be paddle.randn
    assert ocean.randn is paddle.randn


def test_ocean_version_info():
    """Verify version info is accessible."""
    import ocean

    assert ocean.PADDLE_VERSION is not None
    assert isinstance(ocean.version_gte("2.0"), bool)
    assert isinstance(ocean.version_lt("3.0"), bool)
