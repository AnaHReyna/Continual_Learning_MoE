import tensorflow as tf
import tensorflow.keras.layers as layers


class GeoEncoderMLP(tf.keras.Model):
    """
    Local scene-mode embedding.

    Exemplos de modo:
      - straight
      - curve
      - pedestrian-related local context

    Por enquanto ele é treinado só indiretamente pela loss principal.
    Depois podemos adicionar supervisão contrastiva.
    """
    def __init__(self, geo_dim=8, name="geo_encoder"):
        super().__init__(name=name)
        self.geo_dim = int(geo_dim)

        self.net = tf.keras.Sequential([layers.Flatten(),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dense(64, activation="relu"),
                                        layers.Dense(self.geo_dim, activation=None),
                                        ], 
                                        name="geo_encoder_mlp"
                                      )

    def call(self, h, training=False):
        return self.net(h, training=training)
    


class GeoEncoderCrossAttn(tf.keras.Model):
    def __init__(self,
                 geo_dim=8,
                 scene_dim=128,
                 num_heads=2,
                 hidden_dim=128,
                 name="geo_encoder_crossattn"):
        super().__init__(name=name)

        self.scene_dim = int(scene_dim)
        self.geo_dim = int(geo_dim)

        self.query_proj = layers.Dense(self.scene_dim, activation=None, name="geo_query_proj")

        self.map_cross_attn = layers.MultiHeadAttention(num_heads=num_heads,
                                                        key_dim=max(1, self.scene_dim // num_heads),
                                                        output_shape=self.scene_dim,
                                                        name="geo_map_cross_attn",
                                                        )

        self.norm1 = layers.LayerNormalization(name="geo_norm1")
        self.norm2 = layers.LayerNormalization(name="geo_norm2")

        self.ffn = tf.keras.Sequential([layers.Dense(hidden_dim, activation="relu"),
                                        layers.Dense(self.scene_dim, activation=None),
                                       ], 
                                       name="geo_ffn"
                                       )

        self.out_proj = layers.Dense(self.geo_dim, activation=None, name="geo_out_proj")

    def call(self, h, map_tokens=None, training=False):
        q = self.query_proj(h)[:, tf.newaxis, :]   # [B,1,D]

        if map_tokens is None:
            z = tf.squeeze(q, axis=1)
            return self.out_proj(z, training=training)

        attn = self.map_cross_attn(query=q,
                                   key=map_tokens,
                                   value=map_tokens,
                                   training=training,
                                  )

        x = self.norm1(q + attn)
        ff = self.ffn(x, training=training)
        x = self.norm2(x + ff)

        z = tf.squeeze(x, axis=1)   # [B,D]
        z_geo = self.out_proj(z, training=training)
        return z_geo
    

