Quick Start
===========

Installation
-----------

.. code-block:: bash

   pip install -e /path/to/PaddleOcean

Minimal Example
--------------

.. code-block:: python

   import paddle
   import ocean

   # Keras mode
   net = paddle.nn.Sequential(
       paddle.nn.Flatten(),
       paddle.nn.Linear(784, 256),
       paddle.nn.ReLU(),
       paddle.nn.Linear(256, 10),
   )
   model = ocean.Model(net)
   model.prepare(
       optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
       loss=paddle.nn.CrossEntropyLoss(),
       metrics=[paddle.metric.Accuracy()],
   )
   model.fit(train_loader, epochs=10)

   # Lightning mode
   class MyModel(ocean.Model):
       def training_step(self, batch, batch_idx):
           x, y = batch
           loss = paddle.nn.functional.cross_entropy(self(x), y)
           self.log("train_loss", loss)
           return loss
       def configure_optimizers(self):
           return paddle.optimizer.Adam(learning_rate=0.001, parameters=self.parameters())

   model = MyModel()
   trainer = ocean.Trainer(max_epochs=10)
   trainer.fit(model, train_loader)

Key Features
-----------

* **Trainer** — Full training engine with fit/validate/test/predict
* **Model** — Dual-mode: Keras-style (compile+fit) or Lightning-style (hooks)
* **DataModule** — Data lifecycle management
* **Callbacks** — 18 built-in callbacks (checkpoint, early stopping, timer, SWA, ...)
* **Loggers** — 9 loggers (VisualDL, TensorBoard, CSV, Wandb, MLFlow, ...)
* **Strategies** — Single device, DDP, DeepSpeed, FSDP, model parallel
* **Accelerators** — CPU, CUDA, ROCm, XPU, IPU, CustomDevice (天数/昇腾/寒武纪)
* **Gear** — Manual training loop (Fabric-equivalent)
* **Cloud** — AI Studio dataset/model upload/download
* **Compat** — Cross-version PaddlePaddle compatibility (2.4 ~ 3.3)
