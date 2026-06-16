import tensorflow as tf
import tensorflow.keras.layers as layers


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim=128, num_heads=4, ff_dim=256, dropout=0.1, name=None):
        super().__init__(name=name)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads,
                                              key_dim=model_dim // num_heads,
                                              dropout=dropout,
                                              name="mha",
                                             )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(model_dim),
        ], name="ffn")
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        attn_out = self.attn(x, x, attention_mask=mask, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x


class TeacherTaskEncoder(tf.keras.Model):
    """
    Entrada:
      states  : [B, L, state_dim]
      actions : [B, L, action_dim]
      rewards : [B, L, 1]
      mask    : [B, L] opcional (1=válido, 0=padding)

    Saída:
      z_task  : [B, task_emb_dim]
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        window_len,
        model_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        task_emb_dim=16,
        dropout=0.1,
        use_cls_token=True,
        name="teacher_task_encoder",
    ):
        super().__init__(name=name)
        self.window_len = int(window_len)
        self.model_dim = int(model_dim)
        self.task_emb_dim = int(task_emb_dim)
        self.use_cls_token = bool(use_cls_token)

        self.step_proj = tf.keras.Sequential([
            layers.Dense(model_dim, activation="relu"),
            layers.Dense(model_dim, activation=None),
        ], name="step_proj")

        self.pos_emb = layers.Embedding(
            input_dim=window_len + 1,
            output_dim=model_dim,
            name="pos_embedding",
        )

        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, model_dim),
                initializer="random_normal",
                trainable=True,
            )

        self.blocks = [
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                name=f"transformer_block_{i}",
            )
            for i in range(num_layers)
        ]

        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name="final_ln")
        self.out_proj = tf.keras.Sequential([
            layers.Dense(model_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(task_emb_dim, activation=None),
        ], name="task_out_proj")

    def call(self, states, actions, rewards, mask=None, training=False):
        # [B, L, Ds+Da+1]
        x = tf.concat([states, actions, rewards], axis=-1)

        # token por passo
        x = self.step_proj(x, training=training)  # [B, L, D]

        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if self.use_cls_token:
            cls = tf.tile(self.cls_token, [batch_size, 1, 1])  # [B, 1, D]
            x = tf.concat([cls, x], axis=1)                    # [B, L+1, D]
            pos_ids = tf.range(seq_len + 1)[tf.newaxis, :]
            if mask is not None:
                cls_mask = tf.ones((batch_size, 1), dtype=mask.dtype)
                mask = tf.concat([cls_mask, mask], axis=1)
        else:
            pos_ids = tf.range(seq_len)[tf.newaxis, :]

        x = x + self.pos_emb(pos_ids)

        attn_mask = None
        if mask is not None:
            # MultiHeadAttention aceita máscara broadcastable
            attn_mask = mask[:, tf.newaxis, tf.newaxis, :]

        for block in self.blocks:
            x = block(x, training=training, mask=attn_mask)

        x = self.final_norm(x)

        if self.use_cls_token:
            pooled = x[:, 0, :]  # CLS
        else:
            if mask is None:
                pooled = tf.reduce_mean(x, axis=1)
            else:
                mask_f = tf.cast(mask, x.dtype)[..., tf.newaxis]
                pooled = tf.reduce_sum(x * mask_f, axis=1) / (tf.reduce_sum(mask_f, axis=1) + 1e-8)

        z_task = self.out_proj(pooled, training=training)
        z_task = tf.math.l2_normalize(z_task, axis=-1)
        return z_task