from typing import Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from algos.modules.policy import RLEncoder
from common.task_encoder import TaskEncoder
from common.moe_layers import Router, MixtureOfExperts


class StudentMoE(tf.keras.Model):
    def __init__(self,
                 params: Dict[str, Any],
                 state_shape,
                 action_dim: int,
                 max_action: float = 1.0,
                 num_experts: int = 2,
                 task_dim: int = 16,
                 name: str = "student_moe",
                ):

        super().__init__(name=name)

        self.params = dict(params)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.task_dim = int(task_dim)
        self.num_experts = int(num_experts)

        self.use_map = bool(self.params["use_map"] or self.params["use_hier"])
        self.use_vision = bool(self.params.get("use_vision", False))
        self.vision_dim = int(self.params.get("vision_dim", 280))
        self.fusion_type = self.params.get("fusion_type", "cross")

        self.encoder = RLEncoder(state_shape,
                                 units=self.params["units"],
                                 state_input=self.params["state_input"],
                                 lstm=self.params["LSTM"],
                                 trans=False,
                                 cnn_lstm=self.params["cnn_lstm"],
                                 bptt=self.params["bptt"],
                                 ego_surr=self.params["ego_surr"],
                                 neighbours=self.params["neighbours"],
                                 time_step=self.params["time_step"],
                                 debug=False,
                                 make_rotation=self.params["make_rotation"],
                                 use_map=self.params["use_map"],
                                 num_traj=self.params["num_traj"],
                                 cnn=self.params["cnn"],
                                 path_length=self.params["path_length"],
                                 num_heads=self.params["head_num"],
                                 use_hier=self.params["use_hier"],
                                 random_aug=self.params["random_aug"],
                                 no_ego_fut=self.params["no_ego_fut"],
                                 no_neighbor_fut=self.params["no_neighbor_fut"],
                                 carla=self.params["carla"],
                                 use_vision=self.use_vision,
                                 vision_dim=self.vision_dim,
                                 fusion_type=self.fusion_type,
                                )

        self.task_encoder = TaskEncoder(task_dim=self.task_dim)
        self.pre_router = layers.Dense(128, activation="relu", name="pre_router")
        self.router = Router(num_experts=self.num_experts)
        self.moe = MixtureOfExperts(num_experts=self.num_experts, hidden_dim=128, out_dim=128)
        self.out_mean = layers.Dense(self.action_dim, name="student_moe_out_mean")

        self._build_model(state_shape)


    def _build_model(self, state_shape):
        dummy_state = tf.constant(np.zeros((1,) + state_shape, dtype=np.float32))
        dummy_mask = tf.ones((1, state_shape[1]), dtype=tf.float32)

        if self.use_map:
            dummy_map = tf.constant(np.zeros((1, state_shape[0] * 2, self.params["path_length"], 5), dtype=np.float32))
        else:
            dummy_map = None

        dummy_vision = (tf.constant(np.zeros((1, self.vision_dim), dtype=np.float32))
                        if self.use_vision else None
                       )

        _ = self(dummy_state, mask=dummy_mask, map_state=dummy_map, vision=dummy_vision, training=False, return_aux=False,)


    def encode_backbone(self, obs, mask=None, map_state=None, vision=None, training=False):
        feat, _ = self.encoder(obs, mask=mask, test=not training, init_state=None, map_state=map_state, aug=training, vision=vision,)
        if len(feat.shape) == 3:
            feat = tf.reduce_mean(feat, axis=1)
        return feat
    

    def call(self, obs, mask=None, map_state=None, vision=None, training=False, return_aux=False):
        h = self.encode_backbone(obs, mask=mask, map_state=map_state, vision=vision, training=training)
        z = self.task_encoder(h, training=training)

        router_in = tf.concat([h, z], axis=-1)
        router_in = self.pre_router(router_in)

        gate_probs, gate_logits = self.router(router_in, training=training)
        moe_out, expert_outs = self.moe(h, gate_probs, training=training)

        raw_mean = self.out_mean(moe_out)
        action = tf.tanh(raw_mean) * self.max_action

        if return_aux:
            return {"action": action,
                    "raw_mean": raw_mean,
                    "task_embedding": z,
                    "gate_probs": gate_probs,
                    "gate_logits": gate_logits,
                    "expert_outs": expert_outs,
                    "backbone_feat": h,
                    }
        
        return action