from typing import Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from algos.modules.policy import RLEncoder
from common.task_encoder import TaskEncoder
from common.moe_layers import Router, MixtureOfExperts
from stage2.geo_encoder import GeoEncoderMLP, GeoEncoderCrossAttn
from stage2.interaction_encoder import InteractionEncoderMLP, InteractionEncoderCrossAttn


class StudentMoEStage2(tf.keras.Model):
    """
    Stage-2 MoE student with expert expansion.

    num_total_experts = num_old_experts + num_new_experts

    Phase 1:
      - freeze shared backbone
      - freeze old experts
      - train router + new experts + task encoder + output head

    Phase 2:
      - freeze shared backbone
      - freeze router
      - train all experts + task encoder + output head
    """

    def __init__(self,
                 params: Dict[str, Any],
                 state_shape,
                 action_dim: int,
                 max_action: float = 1.0,
                 num_old_experts: int = 2,
                 num_new_experts: int = 1,
                 task_dim: int = 16,
                 geo_dim: int = 8,
                 int_dim: int = 8,
                 geo_type: str = "mlp",
                 interaction_type: str = "mlp",
                 name: str = "student_moe_stage2",
                 use_geo: bool = False,
                 use_int: bool = False,
                ):
        
        super().__init__(name=name)

        self.params = dict(params)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.task_dim = int(task_dim)
        self.geo_dim = int(geo_dim)
        self.int_dim = int(int_dim)
        self.num_old_experts = int(num_old_experts)
        self.num_new_experts = int(num_new_experts)
        self.num_total_experts = self.num_old_experts + self.num_new_experts

        self.use_map = bool(self.params["use_map"] or self.params["use_hier"])
        self.use_vision = bool(self.params.get("use_vision", False))
        self.vision_dim = int(self.params.get("vision_dim", 280))
        self.fusion_type = self.params.get("fusion_type", "cross")
        self.interaction_type = str(interaction_type)
        self.geo_type = str(geo_type)
        self.use_geo = bool(use_geo)
        self.use_int = bool(use_int)

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

        if self.use_geo:
            if self.geo_type == "mlp":
                self.geo_encoder = GeoEncoderMLP(geo_dim=self.geo_dim)
            elif self.geo_type == "cross_attn":
                self.geo_encoder = GeoEncoderCrossAttn(
                    geo_dim=self.geo_dim,
                    scene_dim=self.params["units"],
                    num_heads=self.params["head_num"],
                    hidden_dim=self.params["units"],
                )
            else:
                raise ValueError(f"Unknown geo_type: {self.geo_type}")
        else:
            self.geo_encoder = None

        if self.use_int:
            if self.interaction_type == "mlp":
                self.interaction_encoder = InteractionEncoderMLP(
                    int_dim=self.int_dim,
                    hidden_dim=self.params["units"],
                )
            elif self.interaction_type == "cross_attn":
                self.interaction_encoder = InteractionEncoderCrossAttn(
                    int_dim=self.int_dim,
                    scene_dim=self.params["units"],
                    num_heads=self.params["head_num"],
                    hidden_dim=self.params["units"],
                )
            else:
                raise ValueError(f"Unknown interaction_type: {self.interaction_type}")
        else:
            self.interaction_encoder = None



        self.pre_router = layers.Dense(128, activation="relu", name="pre_router")
        self.router = Router(num_experts=self.num_total_experts)
        self.moe = MixtureOfExperts(num_experts=self.num_total_experts,
                                    hidden_dim=128,
                                    out_dim=128,
                                    )
        self.out_mean = layers.Dense(self.action_dim, name="student_moe_stage2_out_mean")

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
        feat, aux = self.encoder(obs,
                               mask=mask,
                               test=not training,
                               init_state=None,
                               map_state=map_state,
                               aug=training,
                               vision=vision,
                               return_tokens=True,
                               )
        if len(feat.shape) == 3:
            h = tf.reduce_mean(feat, axis=1)
        else:
            h = feat

        hier_tokens = None
        if aux is not None:
            hier_tokens = aux.get("hier_tokens", None)

        return h, hier_tokens
    

    def call(self, obs, mask=None, map_state=None, vision=None, training=False, return_aux=False):

        h, hier_tokens = self.encode_backbone(obs, mask=mask, map_state=map_state, vision=vision, training=training)
        z_task = self.task_encoder(h, training=training)

        actor_tokens = None
        map_tokens = None
        if hier_tokens is not None:
            actor_tokens = hier_tokens.get("actor_tokens", None)
            map_tokens = hier_tokens.get("map_tokens", None)

        parts = [h, z_task]

        if self.use_geo:
            if self.geo_type == "mlp":
                z_geo = self.geo_encoder(h, training=training)
            else:
                z_geo = self.geo_encoder(h, map_tokens=map_tokens, training=training)
            parts.append(z_geo)
        else:
            z_geo = None

        if self.use_int:
            if self.interaction_type == "mlp":
                z_int = self.interaction_encoder(h, training=training)
            else:
                z_int = self.interaction_encoder(h, actor_tokens=actor_tokens, training=training)
            parts.append(z_int)
        else:
            z_int = None

        router_in = tf.concat(parts, axis=-1)
        router_in = self.pre_router(router_in)

        gate_probs, gate_logits = self.router(router_in, training=training)
        moe_out, expert_outs = self.moe(h, gate_probs, training=training)

        raw_mean = self.out_mean(moe_out)
        action = tf.tanh(raw_mean) * self.max_action

        if return_aux:
            return {"action": action,
                    "raw_mean": raw_mean,
                    "task_embedding": z_task,
                    "geo_embedding": z_geo,
                    "interaction_embedding": z_int,
                    "gate_probs": gate_probs,
                    "gate_logits": gate_logits,
                    "expert_outs": expert_outs,
                    "backbone_feat": h,
                    "actor_tokens": actor_tokens,
                    "map_tokens": map_tokens,
                    }
        return action


    def freeze_backbone(self):
        self.encoder.trainable = False


    def freeze_router(self):
        self.pre_router.trainable = False
        self.router.trainable = False


    def unfreeze_router(self):
        self.pre_router.trainable = True
        self.router.trainable = True


    def freeze_old_experts(self):
        for i, expert in enumerate(self.moe.experts):
            if i < self.num_old_experts:
                expert.trainable = False


    def unfreeze_all_experts(self):
        for expert in self.moe.experts:
            expert.trainable = True
            

    def configure_phase(self, phase: int):
        if phase not in [1, 2]:
            raise ValueError(f"Unknown phase: {phase}")

        self.freeze_backbone()
        self.task_encoder.trainable = True

        if self.geo_encoder is not None:
            self.geo_encoder.trainable = True

        if self.interaction_encoder is not None:
            self.interaction_encoder.trainable = True

        self.out_mean.trainable = True

        if phase == 1:
            self.unfreeze_router()
            self.freeze_old_experts()
            for i, expert in enumerate(self.moe.experts):
                if i >= self.num_old_experts:
                    expert.trainable = True

        elif phase == 2:
            self.freeze_router()
            self.unfreeze_all_experts()