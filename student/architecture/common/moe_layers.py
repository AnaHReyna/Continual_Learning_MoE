import tensorflow as tf
import tensorflow.keras.layers as layers


class ExpertMLP(layers.Layer):
    def __init__(self, hidden_dim=128, out_dim=128, name=None):
        super().__init__(name=name)
        self.fc1 = layers.Dense(hidden_dim, activation="relu")
        self.fc2 = layers.Dense(out_dim, activation="relu")

    def call(self, x, training=False):
        return self.fc2(self.fc1(x))


class Router(layers.Layer):
    def __init__(self, num_experts, name="router"):
        super().__init__(name=name)
        self.num_experts = int(num_experts)
        self.logits = layers.Dense(self.num_experts)

    def call(self, x, training=False):
        gate_logits = self.logits(x)
        gate_probs = tf.nn.softmax(gate_logits, axis=-1)
        return gate_probs, gate_logits


class MixtureOfExperts(layers.Layer):
    def __init__(self, num_experts, hidden_dim=128, out_dim=128, name="moe"):
        super().__init__(name=name)
        self.num_experts = int(num_experts)
        self.experts = []
        for i in range(self.num_experts):
            self.experts.append(ExpertMLP(hidden_dim=hidden_dim, out_dim=out_dim, name=f"expert_{i}"))


    def call(self, x, gate_probs, training=False):
        expert_outs = []
        for expert in self.experts:
            out = expert(x, training=training)
            expert_outs.append(out)

        expert_outs = tf.stack(expert_outs, axis=1)
        gate_probs = gate_probs[:, :, tf.newaxis]
        mixed = tf.reduce_sum(expert_outs * gate_probs, axis=1)
        return mixed, expert_outs


def router_balance_loss(gate_probs):
    mean_probs = tf.reduce_mean(gate_probs, axis=0)
    num_experts = tf.cast(tf.shape(mean_probs)[0], tf.float32)
    uniform = tf.ones_like(mean_probs) / tf.maximum(num_experts, 1.0)
    return tf.reduce_sum(tf.square(mean_probs - uniform))


def router_entropy(gate_probs, eps=1e-8):
    ent = -tf.reduce_sum(gate_probs * tf.math.log(gate_probs + eps), axis=-1)
    return tf.reduce_mean(ent)