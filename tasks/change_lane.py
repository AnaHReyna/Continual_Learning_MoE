# tasks/change_lane.py

import os
from collections import deque
import math
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.change_lane import ChangeLane


class ActorConfig:
    def __init__(self, model, transform):
        self.model = model
        self.transform = transform


class ChangeLaneScenarioConfig:
    def __init__(self, trigger_point, route, target_speed_kmh):
        self.trigger_points = [trigger_point]
        self.route = route
        self.target_speed_kmh = target_speed_kmh
        self.name = "ChangeLane"
        self.town = None
        self.weather = carla.WeatherParameters.ClearNoon
        self.friction = None
        self.other_actors = [ActorConfig("vehicle.tesla.model3", trigger_point),
                             ActorConfig("vehicle.volkswagen.t2", trigger_point),
                            ]


class ChangeLaneTask:
    name = "change_lane"

    def __init__(self,
                 curriculum_level=0,
                 auto_curriculum=True,
                 max_level=3,
                 window_size=50,
                 promote_threshold=0.75,
                 demote_threshold=0.45,
                 cooldown_episodes=25,
                 allow_demotion=True,
                ):
        
        self.curriculum_level = curriculum_level
        self.auto_curriculum = auto_curriculum
        self.max_level = max_level
        self.window_size = window_size
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.cooldown_episodes = cooldown_episodes
        self.allow_demotion = allow_demotion

        self.recent_success = deque(maxlen=window_size)
        self.episodes_since_change = 0
        self.trigger_range = (0.05, 0.15)

    # def _log_bad_route(self, env, reason):
    #     os.makedirs("bad_change_lane_routes", exist_ok=True)

    #     with open("bad_change_lane_routes/bad_routes.txt", "a") as f:
    #         f.write(
    #             f"name={getattr(env, 'route_name', 'unknown')}, "
    #             f"id={getattr(env, 'route_id', 'unknown')}, "
    #             f"length={env._route_total_length:.2f}, "
    #             f"waypoints={len(env.route_waypoints)}, "
    #             f"reason={reason}\n"
    #         )

    def _apply_curriculum(self, env):
        if self.curriculum_level == 0:
            env.cfg.num_npc_vehicles = 0
            env.cfg.target_speed_kmh = 20.0
        elif self.curriculum_level == 1:
            env.cfg.num_npc_vehicles = 0
            env.cfg.target_speed_kmh = 23.0
        elif self.curriculum_level == 2:
            env.cfg.num_npc_vehicles = 1
            env.cfg.target_speed_kmh = 26.0
        else:
            env.cfg.num_npc_vehicles = 1
            env.cfg.target_speed_kmh = 30.0

        self.trigger_range = (0.05, 0.15)

        print(
            f"[TASK ChangeLane] level={self.curriculum_level} "
            f"num_npc={env.cfg.num_npc_vehicles} "
            f"target_speed={env.cfg.target_speed_kmh} "
            f"trigger_range={self.trigger_range}"
        )

    def configure_env(self, cfg):
        cfg.num_npc_vehicles = 0
        cfg.target_speed_kmh = 20.0
        return cfg

    def _spawn_simple_blocking_vehicle(self, env):
        if env.world is None or env.map is None or len(env.route_waypoints) < 2:
            return False

        bp = env.world.get_blueprint_library().find("vehicle.volkswagen.t2")

        idx_min = int(0.45 * len(env.route_waypoints))
        idx_max = int(0.55 * len(env.route_waypoints))

        idx_min = max(1, min(idx_min, len(env.route_waypoints) - 1))
        idx_max = max(idx_min + 1, min(idx_max, len(env.route_waypoints) - 1))

        for _ in range(10):
            idx = env._rng.randint(idx_min, idx_max)
            base_tf = env.route_waypoints[idx]

            wp = env.map.get_waypoint(
                base_tf.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp is None or wp.is_junction:
                continue

            spawn_tf = carla.Transform(
                carla.Location(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z + 0.6,
                ),
                wp.transform.rotation,
            )

            vehicle = env.world.try_spawn_actor(bp, spawn_tf)

            if vehicle is None:
                continue

            vehicle.set_simulate_physics(True)
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
            )

            env.actor_handles.append(vehicle)
            env.npc_vehicles.append(vehicle)
            env.change_lane_blocking_vehicle = vehicle

            # print(
            #     "[ChangeLaneTask] fallback blocking vehicle spawned",
            #     "idx=", idx,
            #     "loc=", spawn_tf.location,
            #     "yaw=", spawn_tf.rotation.yaw,
            # )

            return True

        # print("[ChangeLaneTask] fallback blocking vehicle failed")
        return False

    def on_reset(self, env):
        self._apply_curriculum(env)

        CarlaDataProvider.set_client(env.client)
        CarlaDataProvider.set_world(env.world)
        CarlaDataProvider.set_traffic_manager_port(env.cfg.traffic_manager_port)
        CarlaDataProvider.register_actor(env.ego)

        env.scenario = None
        env.scenario_tree = None
        env.scenario_criteria = []
        env.skip_episode = False
        env.change_lane_blocking_vehicle = None

        fraction = env._rng.uniform(*self.trigger_range)
        idx = int(fraction * len(env.route_waypoints))
        idx = max(1, min(idx, len(env.route_waypoints) - 1))
        trigger_point = env.route_waypoints[idx]

        env.route_name = getattr(
            env,
            "current_route_name",
            getattr(env, "route_name", "unknown"),
        )
        env.route_id = getattr(
            env,
            "current_route_id",
            getattr(env, "route_id", "unknown"),
        )

        # if env._route_total_length < 60.0:
        #     self._log_bad_route(env, "short_route")
        #     env.skip_episode = True
        #     return

        trigger_wp = env.map.get_waypoint(
            trigger_point.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if trigger_wp is None or trigger_wp.is_junction:
            env.skip_episode = True
            return

        has_same_direction_lane = False

        for side_wp in [trigger_wp.get_left_lane(), trigger_wp.get_right_lane()]:
            if side_wp is None:
                continue

            if side_wp.lane_type != carla.LaneType.Driving:
                continue

            yaw_diff = abs(
                (trigger_wp.transform.rotation.yaw - side_wp.transform.rotation.yaw + 180.0)
                % 360.0 - 180.0
            )

            if yaw_diff < 25.0:
                has_same_direction_lane = True
                break

        if not has_same_direction_lane:
            env.skip_episode = True
            return

        config = ChangeLaneScenarioConfig(
            trigger_point=trigger_wp.transform,
            route=env.route_dense,
            target_speed_kmh=env.cfg.target_speed_kmh,
        )

        # try:
        scenario = ChangeLane(world=env.world,
                              ego_vehicles=[env.ego],
                              config=config,
                              randomize=False,
                              debug_mode=False,
                              criteria_enable=True,
                              timeout=60,
                             )

        # except Exception as e:
        #     self._log_bad_route(env, str(e))
        #     env.skip_episode = False
        #     env.scenario = None
        #     env.scenario_tree = None
        #     env.scenario_criteria = []

        #     self._spawn_simple_blocking_vehicle(env)
        #     return

        num_actors = len(getattr(scenario, "other_actors", []))
        # print(f"[ChangeLaneTask] scenario actors={num_actors}")

        if num_actors < 2:
            # print(
            #     "[ChangeLaneTask] WARNING: could not spawn Tesla/VW, "
            #     "using simple blocking vehicle"
            # )

            try:
                scenario.remove_all_actors()
            except Exception:
                pass

            env.skip_episode = False
            env.scenario = None
            env.scenario_tree = None
            env.scenario_criteria = []

            self._spawn_simple_blocking_vehicle(env)
            return

        env.scenario = scenario
        env.scenario_tree = scenario.scenario.scenario_tree
        env.scenario_criteria = scenario.criteria_list

    def after_tick(self, env):
        if env.scenario_tree is not None:
            env.scenario_tree.tick_once()

    def _scenario_status(self, env):
        if env.scenario_tree is None:
            return False, False, {}

        status = env.scenario_tree.status

        if str(status) == "Status.SUCCESS" or status == "SUCCESS":
            return True, False, {"scenario_event": "success"}

        if str(status) == "Status.FAILURE" or status == "FAILURE":
            return False, True, {"scenario_event": "failure"}

        return False, False, {}

    def _actor_in_ego_frame(self, env, actor):
        ego_tf = env.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        actor_loc = actor.get_location()

        dx = float(actor_loc.x - ego_loc.x)
        dy = float(actor_loc.y - ego_loc.y)

        c = math.cos(ego_yaw)
        s = math.sin(ego_yaw)

        x_local = c * dx + s * dy
        y_local = -s * dx + c * dy

        return x_local, y_local

    def _get_change_lane_metrics(self, env):
        result = {
            "tesla_front_dist": float("inf"),
            "vw_front_dist": float("inf"),
            "vw_x_local": float("inf"),
            "vw_lateral_dist": float("inf"),
            "vw_ttc": float("inf"),
            "front_vehicle_dist": float("inf"),
            "front_vehicle_ttc": float("inf"),
            "ego_changed_lane": False,
        }

        if env.ego is None:
            return result

        ego_speed = env._kmh(env.ego.get_velocity()) / 3.6

        actors = []

        for actor in env.npc_vehicles:
            if actor is not None and actor.is_alive:
                actors.append(actor)

        if env.scenario is not None and hasattr(env.scenario, "other_actors"):
            for actor in env.scenario.other_actors:
                if actor is not None and actor.is_alive:
                    actors.append(actor)

        best_front_x = float("inf")
        best_front_ttc = float("inf")

        for actor in actors:
            x_local, y_local = self._actor_in_ego_frame(env, actor)

            actor_speed = env._kmh(actor.get_velocity()) / 3.6
            rel_speed = max(ego_speed - actor_speed, 0.0)

            if rel_speed > 1e-3:
                ttc = x_local / rel_speed
            else:
                ttc = float("inf")

            if x_local > 0.0 and abs(y_local) < 3.5:
                if x_local < best_front_x:
                    best_front_x = x_local
                    best_front_ttc = ttc

        result["front_vehicle_dist"] = best_front_x
        result["front_vehicle_ttc"] = best_front_ttc

        if env.scenario is not None and hasattr(env.scenario, "other_actors"):
            scenario_actors = env.scenario.other_actors

            if len(scenario_actors) >= 1:
                tesla = scenario_actors[0]
                if tesla is not None and tesla.is_alive:
                    x_local, _ = self._actor_in_ego_frame(env, tesla)
                    if x_local > 0.0:
                        result["tesla_front_dist"] = x_local

            if len(scenario_actors) >= 2:
                vw = scenario_actors[1]
                if vw is not None and vw.is_alive:
                    x_local, y_local = self._actor_in_ego_frame(env, vw)

                    result["vw_x_local"] = x_local
                    result["vw_lateral_dist"] = abs(y_local)

                    if x_local > 0.0:
                        result["vw_front_dist"] = x_local

                        vw_speed = env._kmh(vw.get_velocity()) / 3.6
                        rel_speed = max(ego_speed - vw_speed, 0.0)

                        if rel_speed > 1e-3:
                            result["vw_ttc"] = x_local / rel_speed

        blocking_vehicle = getattr(env, "change_lane_blocking_vehicle", None)

        if blocking_vehicle is not None and blocking_vehicle.is_alive:
            x_local, y_local = self._actor_in_ego_frame(env, blocking_vehicle)

            result["vw_x_local"] = x_local
            result["vw_lateral_dist"] = abs(y_local)

            if x_local > 0.0:
                result["vw_front_dist"] = x_local

                blocking_speed = env._kmh(blocking_vehicle.get_velocity()) / 3.6
                rel_speed = max(ego_speed - blocking_speed, 0.0)

                if rel_speed > 1e-3:
                    result["vw_ttc"] = x_local / rel_speed

                if x_local < result["front_vehicle_dist"]:
                    result["front_vehicle_dist"] = x_local
                    result["front_vehicle_ttc"] = result["vw_ttc"]

        lateral_error, _, _, _ = env._compute_route_errors()
        result["ego_changed_lane"] = abs(lateral_error) > 2.0

        return result

    def compute_reward_done(self, env, info):
        if getattr(env, "skip_episode", False):
            return 0.0, True, {
                "finish": False,
                "scenario_success": False,
                "scenario_failure": False,
                "done_reason": "skip_invalid_lane_change_route",
                "task_name": self.name,
                "skip_episode": True,
            }

        scenario_success, scenario_failure, scenario_info = self._scenario_status(env)
        cl_info = self._get_change_lane_metrics(env)

        route_finish = info["route_finish"]
        collision = info["collision"]
        stuck = info["stuck"]

        lateral_error = abs(info["lateral_error"])

        front_dist = cl_info["front_vehicle_dist"]
        front_ttc = cl_info["front_vehicle_ttc"]
        vw_x_local = cl_info["vw_x_local"]
        vw_lateral_dist = cl_info["vw_lateral_dist"]
        ego_changed_lane = cl_info["ego_changed_lane"]

        obstacle_close = 0.0 < front_dist < 25.0
        obstacle_danger = (0.0 < front_dist < 14.0) or (front_ttc < 3.5)
        far_from_obstacle = front_dist > 30.0

        safe_to_return = vw_x_local < -8.0
        back_to_route = lateral_error < 1.5

        episode_step = getattr(env, "step_count", 0)

        if episode_step < 5:
            off_route = False
        else:
            off_route = lateral_error > 8.0

        finish = (
            route_finish
            and back_to_route
            and not collision
            and not scenario_failure
        )

        done = bool(finish or collision or stuck or scenario_failure or off_route)

        if finish:
            done_reason = "finish"
        elif collision:
            done_reason = "collision"
        elif scenario_failure:
            done_reason = "scenario_failure"
        elif off_route:
            done_reason = "off_route"
        elif stuck:
            done_reason = "stuck"
        else:
            done_reason = None

        # if done:
        #     print(
        #         "[DONE DEBUG]",
        #         "route_finish=", route_finish,
        #         "back_to_route=", back_to_route,
        #         "lateral_error=", lateral_error,
        #         "collision=", collision,
        #         "vw_x_local=", vw_x_local,
        #         "vw_lateral_dist=", vw_lateral_dist,
        #         "safe_to_return=", safe_to_return,
        #         "reason=", done_reason,
        #     )

        reward = 0.0

        reward += 3.0 * info["progress_delta"]
        reward += 0.03 * info["dist_delta"]

        reward -= 0.015 * abs(info["heading_error"])
        reward -= 0.03 * abs(info["control_steer"])
        reward -= 0.08 * info["steer_delta"]

        if far_from_obstacle and not ego_changed_lane:
            reward += 0.05

        if far_from_obstacle and ego_changed_lane:
            reward -= 0.30

        if obstacle_close:
            if not ego_changed_lane:
                reward -= 1.2

            if ego_changed_lane and lateral_error < 3.2:
                reward -= 1.8

            if ego_changed_lane and 3.8 <= lateral_error <= 5.0:
                reward += 1.2

            if ego_changed_lane and lateral_error > 5.3:
                reward -= 2.0

        if obstacle_danger and not ego_changed_lane:
            reward -= 2.0

        if obstacle_danger and ego_changed_lane and lateral_error < 3.6:
            reward -= 2.8

        if lateral_error > 5.8:
            reward -= 3.0

        if lateral_error > 6.8:
            reward -= 6.0

        near_vw_longitudinal = -2.0 < vw_x_local < 10.0
        near_vw_lateral = vw_lateral_dist < 5.0

        if near_vw_longitudinal and near_vw_lateral:
            reward -= 3.0

        if front_ttc < 4.0:
            reward -= 0.08 * (4.0 - front_ttc)

        # Não pode voltar para a rota enquanto o VW ainda não ficou bem para trás.
        if not safe_to_return and back_to_route:
            reward -= 3.0

        # Só recompensa voltar quando o VW já ficou para trás.
        if safe_to_return and back_to_route:
            reward += 2.0

        if safe_to_return and lateral_error > 2.5:
            reward -= 1.0

        if scenario_success:
            reward += 0.5

        if finish:
            reward += 2.0

        if collision:
            reward -= 5.0

        if scenario_failure:
            reward -= 2.0

        if off_route:
            reward -= 3.0

        if stuck:
            reward -= 0.5

        task_info = {
            "finish": finish,
            "scenario_success": scenario_success,
            "scenario_failure": scenario_failure,
            "done_reason": done_reason,
            "task_name": self.name,
            "curriculum_level": self.curriculum_level,
        }

        task_info.update(cl_info)
        task_info.update(scenario_info)

        return reward, done, task_info

    def record_episode_result(self, success):
        if not self.auto_curriculum:
            return

        if success is None:
            return

        self.recent_success.append(1.0 if success else 0.0)
        self.episodes_since_change += 1

        if len(self.recent_success) < self.recent_success.maxlen:
            return

        if self.episodes_since_change < self.cooldown_episodes:
            return

        success_rate = sum(self.recent_success) / len(self.recent_success)

        if success_rate >= self.promote_threshold and self.curriculum_level < self.max_level:
            self.curriculum_level += 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            # print(f"[Curriculum] upgraded change_lane to level {self.curriculum_level}")
            return

        if self.allow_demotion and success_rate <= self.demote_threshold and self.curriculum_level > 0:
            self.curriculum_level -= 1
            self.episodes_since_change = 0
            self.recent_success.clear()
            # print(f"[Curriculum] change_lane dropped to level {self.curriculum_level}")