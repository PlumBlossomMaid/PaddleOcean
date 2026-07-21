"""Tests for manual optimization precision support.

Covers:
- training_step runs inside the precision plugin's forward_context (AMP auto_cast)
- manual_backward routes through the precision plugin (GradScaler.scale)
- OceanOptimizer.step routes through the plugin in manual mode (scaler.step/update)
- Model.clip_gradients unscales before clipping under AMP
- fp32 manual path unchanged
"""

import paddle

import ocean
from ocean.model import Model


class ManualAMPModel(Model):
    """GAN-style dual-optimizer model exercising manual_backward + clip_gradients."""

    def __init__(self, clip: bool = True):
        super().__init__()
        self.automatic_optimization = False
        self.clip = clip
        self.gen = paddle.nn.Linear(8, 8)
        self.disc = paddle.nn.Linear(8, 1)
        self.saw_float16 = False

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        fake = self.gen(batch)
        if fake.dtype == paddle.float16:
            self.saw_float16 = True
        loss_g = fake.mean()
        opt_g.clear_grad()
        self.manual_backward(loss_g)
        if self.clip:
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt_g.step()

        loss_d = -self.disc(fake.detach().astype("float32")).mean()
        opt_d.clear_grad()
        self.manual_backward(loss_d)
        if self.clip:
            self.clip_gradients(opt_d, gradient_clip_val=0.5)
        opt_d.step()
        return loss_g

    def configure_optimizers(self):
        return [
            paddle.optimizer.Adam(learning_rate=1e-3, parameters=self.gen.parameters()),
            paddle.optimizer.Adam(learning_rate=1e-3, parameters=self.disc.parameters()),
        ]


def _loader(n=6, dim=8):
    data = paddle.randn([n, dim])

    class _DS(paddle.io.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return data[i]

    return paddle.io.DataLoader(_DS(), batch_size=2, shuffle=False)


def _trainer(precision="16-mixed", max_steps=3):
    return ocean.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        precision=precision,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
    )


def test_manual_amp_training_updates_weights():
    """manual + 16-mixed: weights of both G and D move, values stay finite."""
    model = ManualAMPModel()
    w_g = model.gen.weight.clone()
    w_d = model.disc.weight.clone()
    trainer = _trainer()
    trainer.fit(model, _loader())

    d_gen = float((model.gen.weight - w_g).abs().max())
    d_disc = float((model.disc.weight - w_d).abs().max())
    assert d_gen > 0, "generator weights did not move under manual+AMP"
    assert d_disc > 0, "discriminator weights did not move under manual+AMP"
    assert bool(paddle.isfinite(model.gen.weight).all()), "non-finite generator weights"
    # G and D each stepped once per batch
    assert trainer.optimizer_step == 2 * trainer.dataloader_step


def test_manual_training_step_runs_in_forward_context():
    """training_step must be wrapped by the precision plugin's forward_context.

    (On CPU paddle's auto_cast is a silent no-op, so we assert the routing —
    the plugin context is entered — rather than observing float16 tensors.)
    """
    calls = []
    model = ManualAMPModel(clip=False)
    trainer = _trainer(max_steps=1)

    from ocean.plugins.precision.amp import MixedPrecision

    orig = MixedPrecision.forward_context

    def probe(self):
        calls.append("forward_context")
        return orig(self)

    MixedPrecision.forward_context = probe
    try:
        trainer.fit(model, _loader())
    finally:
        MixedPrecision.forward_context = orig
    assert "forward_context" in calls, "training_step did not run inside precision forward_context"


def test_manual_backward_routes_through_plugin():
    """manual_backward must call the precision plugin's backward (GradScaler.scale).

    (Scaling itself is a no-op on CPU, so assert the routing.)
    """
    calls = []
    model = ManualAMPModel(clip=False)
    trainer = _trainer(max_steps=1)

    from ocean.plugins.precision.amp import MixedPrecision

    orig = MixedPrecision.backward

    def probe(self, tensor, m, *args, **kwargs):
        calls.append("backward")
        return orig(self, tensor, m, *args, **kwargs)

    MixedPrecision.backward = probe
    try:
        trainer.fit(model, _loader())
    finally:
        MixedPrecision.backward = orig
    # G + D each backward once in the single step
    assert calls.count("backward") == 2, (
        f"manual_backward bypassed the precision plugin ({calls.count('backward')} calls)"
    )


def test_manual_optimizer_step_routes_through_plugin():
    """OceanOptimizer.step in manual mode must call plugin.optimizer_step (scaler.step/update)."""
    calls = []
    model = ManualAMPModel(clip=False)
    trainer = _trainer(max_steps=1)

    from ocean.plugins.precision.amp import MixedPrecision

    orig = MixedPrecision.optimizer_step

    def probe(self, optimizer, **kwargs):
        calls.append("optimizer_step")
        return orig(self, optimizer, **kwargs)

    MixedPrecision.optimizer_step = probe
    try:
        trainer.fit(model, _loader())
    finally:
        MixedPrecision.optimizer_step = orig
    assert calls.count("optimizer_step") == 2, (
        f"manual opt.step() bypassed the precision plugin ({calls.count('optimizer_step')} calls)"
    )


def test_manual_clip_gradients_unscales():
    """clip_gradients under AMP unscales first, so post-clip norm <= clip_val."""
    captured = {}

    class ProbeModel(ManualAMPModel):
        def training_step(self, batch, batch_idx):
            opt_g, opt_d = self.optimizers()
            loss_g = self.gen(batch).mean()
            opt_g.clear_grad()
            self.manual_backward(loss_g)
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            if "post_clip" not in captured:
                total = paddle.zeros([1])
                for p in self.parameters():
                    if p.grad is not None:
                        total += (p.grad.astype("float32") ** 2).sum()
                captured["post_clip"] = float(total.sqrt())
            opt_g.step()

            loss_d = -self.disc(self.gen(batch).detach().astype("float32")).mean()
            opt_d.clear_grad()
            self.manual_backward(loss_d)
            opt_d.step()
            return loss_g

    model = ProbeModel()
    trainer = _trainer(max_steps=1)
    trainer.fit(model, _loader())
    assert captured["post_clip"] <= 0.5 + 1e-4, (
        f"post-clip grad norm {captured['post_clip']} exceeds clip_val — unscale missing"
    )


def test_manual_fp32_path_unchanged():
    """fp32 manual optimization still works and counts steps correctly."""
    model = ManualAMPModel()
    trainer = _trainer(precision="32", max_steps=2)
    trainer.fit(model, _loader())
    assert trainer.optimizer_step == 2 * trainer.dataloader_step
    assert bool(paddle.isfinite(model.gen.weight).all())
