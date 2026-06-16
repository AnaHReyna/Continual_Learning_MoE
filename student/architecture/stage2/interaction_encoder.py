import tensorflow as tf
import tensorflow.keras.layers as layers

class InteractionEncoderMLP(tf.keras.Model):

    def __init__(self, int_dim=8, hidden_dim=64, name="interaction_encoder"):
        super().__init__(name=name)
        self.fc1 = layers.Dense(hidden_dim, activation="relu")
        self.fc2 = layers.Dense(int_dim, activation=None)

    def call(self, h, training=False):
        x = self.fc1(h, training=training)
        z_int = self.fc2(x, training=training)
        return z_int
    


class InteractionEncoderCrossAttn(tf.keras.Model):
    def __init__(self,
                 int_dim=8,
                 scene_dim=128,
                 num_heads=2,
                 hidden_dim=128,
                 name="interaction_encoder_crossattn"):
        super().__init__(name=name)

        self.scene_dim = int(scene_dim)
        self.int_dim = int(int_dim)

        self.query_proj = layers.Dense(self.scene_dim, activation=None, name="int_query_proj")

        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=max(1, self.scene_dim // num_heads),
                                                    output_shape=self.scene_dim,
                                                    name="int_cross_attn",
                                                    )

        self.norm1 = layers.LayerNormalization(name="int_norm1")
        self.norm2 = layers.LayerNormalization(name="int_norm2")

        self.ffn = tf.keras.Sequential([layers.Dense(hidden_dim, activation="relu"),
                                        layers.Dense(self.scene_dim, activation=None),
                                        ], 
                                        name="int_ffn"
                                        )

        self.out_proj = layers.Dense(self.int_dim, activation=None, name="int_out_proj")

    def call(self, h, actor_tokens=None, training=False):
        """
        h:            [B, D]
        actor_tokens: [B, Na, D]
        """
        q = self.query_proj(h)[:, tf.newaxis, :]   # [B, 1, D]

        if actor_tokens is None:
            z = tf.squeeze(q, axis=1)
            return self.out_proj(z, training=training)

        attn = self.cross_attn(query=q,
                               key=actor_tokens,
                               value=actor_tokens,
                               training=training,
                              )  # [B, 1, D]

        x = self.norm1(q + attn)
        ff = self.ffn(x, training=training)
        x = self.norm2(x + ff)

        z = tf.squeeze(x, axis=1)   # [B, D]
        z_int = self.out_proj(z, training=training)
        return z_int