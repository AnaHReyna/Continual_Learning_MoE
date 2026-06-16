import tensorflow as tf
import tensorflow.keras.layers as layers


class TaskEncoder(tf.keras.Model):
    """Latent task embedding inferred from the observation sequence."""

    def __init__(self, task_dim=16, name="task_encoder"):
        super().__init__(name=name)
        self.task_dim = int(task_dim)
        self.net = tf.keras.Sequential([layers.Flatten(),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dense(64, activation="relu"),
                                        layers.Dense(self.task_dim, activation=None),
                                        ], 
                                        name="task_encoder_mlp"
                                       )

    def call(self, h, training=False):
        return self.net(h, training=training)