# # # # import math
# # # # import queue
# # # # import random
# # # # import weakref
# # # # from typing import Dict, List, Optional, Tuple

# # # # import carla
# # # # import cv2
# # # # import gymnasium as gym
# # # # import numpy as np
# # # # from gymnasium import spaces

# # # # from srunner.tools.route_parser import RouteParser
# # # # from srunner.tools.route_manipulation import interpolate_trajectory


# # # # class EnvConfig:
# # # #     host = "127.0.0.1"
# # # #     port = 2000
# # # #     traffic_manager_port = 8000
# # # #     timeout = 10.0

# # # #     route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
# # # #     scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"
# # # #     route_id = None      # All episodes will use the first route in the file if route_id is not None. Otherwise, a random route from the file will be chosen for each episode.
# # # #     # route_id = "20"    # route_id can also be set to a specific value to always use the same route (useful for debugging)
# # # #     # route_towns = ["Town01", "Town03", "Town05"]
# # # #     route_town = "Town05"

# # # #     sync = True
# # # #     fixed_delta_seconds = 0.05

# # # #     max_episode_steps = 500
# # # #     target_speed_kmh = 30.0

# # # #     ego_filter = "vehicle.lincoln.mkz_2017"
# # # #     seed = 42

# # # #     render_rgb_camera = True

# # # #     spectator_follow = True
# # # #     spectator_height_m = 40.0
# # # #     spectator_rotate_with_ego = False

# # # #     show_bev = True
# # # #     bev_width = 800
# # # #     bev_height = 800
# # # #     bev_fov = 90
# # # #     bev_height_m = 35.0


# # # # class CollisionSensor:
# # # #     def __init__(self, parent_actor: carla.Actor):
# # # #         self.sensor = None
# # # #         self.history = []
# # # #         self._parent = parent_actor

# # # #         world = self._parent.get_world()
# # # #         bp = world.get_blueprint_library().find("sensor.other.collision")
# # # #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# # # #         weak_self = weakref.ref(self)
# # # #         self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

# # # #     @staticmethod
# # # #     def _on_collision(weak_self, event):
# # # #         self = weak_self()
# # # #         if not self:
# # # #             return

# # # #         impulse = event.normal_impulse
# # # #         intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
# # # #         self.history.append((event.frame, intensity))

# # # #     def clear(self):
# # # #         self.history.clear()

# # # #     def destroy(self):
# # # #         if self.sensor is not None:
# # # #             self.sensor.stop()
# # # #             self.sensor.destroy()
# # # #             self.sensor = None


# # # # class LaneInvasionSensor:
# # # #     def __init__(self, parent_actor: carla.Actor):
# # # #         self.sensor = None
# # # #         self.count = 0
# # # #         self._parent = parent_actor

# # # #         world = self._parent.get_world()
# # # #         bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
# # # #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# # # #         weak_self = weakref.ref(self)
# # # #         self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

# # # #     @staticmethod
# # # #     def _on_invasion(weak_self, event):
# # # #         self = weak_self()
# # # #         if not self:
# # # #             return
# # # #         self.count += 1

# # # #     def clear(self):
# # # #         self.count = 0

# # # #     def destroy(self):
# # # #         if self.sensor is not None:
# # # #             self.sensor.stop()
# # # #             self.sensor.destroy()
# # # #             self.sensor = None


# # # # class CameraSensor:
# # # #     def __init__(self, parent_actor: carla.Actor, width=640, height=360):
# # # #         self.sensor = None
# # # #         self.queue = queue.Queue()
# # # #         self.width = width
# # # #         self.height = height

# # # #         world = parent_actor.get_world()
# # # #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# # # #         bp.set_attribute("image_size_x", str(width))
# # # #         bp.set_attribute("image_size_y", str(height))
# # # #         bp.set_attribute("fov", "90")
# # # #         bp.set_attribute("sensor_tick", "0.0")

# # # #         transform = carla.Transform(carla.Location(x=1.5, z=2.4))
# # # #         self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
# # # #         self.sensor.listen(self.queue.put)

# # # #     def get_latest(self) -> Optional[np.ndarray]:
# # # #         img = None
# # # #         while not self.queue.empty():
# # # #             img = self.queue.get()

# # # #         if img is None:
# # # #             return None

# # # #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# # # #         array = array.reshape((self.height, self.width, 4))
# # # #         array = array[:, :, :3]
# # # #         return array

# # # #     def destroy(self):
# # # #         if self.sensor is not None:
# # # #             self.sensor.stop()
# # # #             self.sensor.destroy()
# # # #             self.sensor = None


# # # # class BirdEyeCamera:
# # # #     """
# # # #     Câmera top-down presa ao ego.
# # # #     Como está attach_to=parent_actor com AttachmentType.Rigid,
# # # #     ela acompanha a posição e a rotação do carro.
# # # #     """

# # # #     def __init__(self, parent_actor: carla.Actor, width=800, height=800, fov=90, z=35.0):
# # # #         self.sensor = None
# # # #         self.queue = queue.Queue()
# # # #         self.width = width
# # # #         self.height = height

# # # #         world = parent_actor.get_world()
# # # #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# # # #         bp.set_attribute("image_size_x", str(width))
# # # #         bp.set_attribute("image_size_y", str(height))
# # # #         bp.set_attribute("fov", str(fov))
# # # #         bp.set_attribute("sensor_tick", "0.0")

# # # #         transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=z),
# # # #                                     carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
# # # #                                     )

# # # #         self.sensor = world.spawn_actor(bp,
# # # #                                         transform,
# # # #                                         attach_to=parent_actor,
# # # #                                         attachment_type=carla.AttachmentType.Rigid,
# # # #                                         )
        
# # # #         self.sensor.listen(self.queue.put)

# # # #     def get_latest(self) -> Optional[np.ndarray]:
# # # #         img = None
# # # #         while not self.queue.empty():
# # # #             img = self.queue.get()

# # # #         if img is None:
# # # #             return None

# # # #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# # # #         array = array.reshape((self.height, self.width, 4))
# # # #         array = array[:, :, :3]
# # # #         return array

# # # #     def destroy(self):
# # # #         if self.sensor is not None:
# # # #             self.sensor.stop()
# # # #             self.sensor.destroy()
# # # #             self.sensor = None


# # # # class CarlaRouteEnv(gym.Env):

# # # #     def __init__(self, cfg: EnvConfig):
# # # #         super().__init__()
# # # #         self.cfg = cfg
# # # #         self._rng = random.Random(cfg.seed)

# # # #         self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
# # # #                                         high=np.array([1.0, 1.0], dtype=np.float32),
# # # #                                         dtype=np.float32,
# # # #                                         )

# # # #         self.observation_space = spaces.Box(low=-np.inf,
# # # #                                             high=np.inf,
# # # #                                             shape=(14,),
# # # #                                             dtype=np.float32,
# # # #                                             )

# # # #         self.client: Optional[carla.Client] = None
# # # #         self.world: Optional[carla.World] = None
# # # #         self.map: Optional[carla.Map] = None
# # # #         self.traffic_manager = None

# # # #         self.route_configs = []
# # # #         self.route_config = None
# # # #         self.route_dense: List[Tuple[carla.Transform, object]] = []
# # # #         self.route_waypoints: List[carla.Transform] = []
# # # #         self.route_index = 0

# # # #         self.ego: Optional[carla.Vehicle] = None
# # # #         self.collision_sensor: Optional[CollisionSensor] = None
# # # #         self.lane_sensor: Optional[LaneInvasionSensor] = None
# # # #         self.camera_sensor: Optional[CameraSensor] = None
# # # #         self.bev_camera: Optional[BirdEyeCamera] = None

# # # #         self.actor_handles: List[carla.Actor] = []

# # # #         self.step_count = 0
# # # #         self.last_progress = 0.0
# # # #         self.last_location: Optional[carla.Location] = None
# # # #         self.stuck_steps = 0
# # # #         self.prev_dist_to_goal = None

# # # #         self._bev_window_created = False


# # # #     def _connect(self):
# # # #         self.client = carla.Client(self.cfg.host, self.cfg.port)
# # # #         self.client.set_timeout(self.cfg.timeout)
# # # #         self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)
# # # #         # self.traffic_manager = None


# # # #     def _load_route_configs(self):
# # # #         self.route_configs = RouteParser.parse_routes_file(self.cfg.route_file,
# # # #                                                             self.cfg.scenario_file,
# # # #                                                             single_route=self.cfg.route_id,
# # # #                                                             )
        
# # # #         if not self.route_configs:
# # # #             raise RuntimeError("No route found in the XML file.")
        

# # # #     def _choose_route(self):
# # # #         if self.cfg.route_id is not None:
# # # #             self.route_config = self.route_configs[0]
# # # #             # print('====== ANA ======', self.route_config.town)

# # # #         route_town = getattr(self.cfg, "route_town", None)

# # # #         if route_town is not None:
# # # #             valid_routes = []
# # # #             for rc in self.route_configs:
# # # #                 if rc.town == self.cfg.route_town:
# # # #                     valid_routes.append(rc)

# # # #             if not valid_routes: 
# # # #                 raise RuntimeError(f"No route found for town {self.cfg.route_town}") 

# # # #             self.route_config = self._rng.choice(valid_routes)  

# # # #         else:
# # # #             self.route_config = self._rng.choice(self.route_configs)
                


# # # #     def _load_world_for_route(self):
# # # #         assert self.client is not None
# # # #         assert self.route_config is not None

# # # #         town = self.route_config.town
# # # #         self.world = self.client.load_world(town)
# # # #         self.map = self.world.get_map()

# # # #         if self.cfg.sync:
# # # #             settings = self.world.get_settings()
# # # #             settings.synchronous_mode = True
# # # #             settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
# # # #             self.world.apply_settings(settings)

# # # #             self.client.reload_world(False)
# # # #             self.world = self.client.get_world()
# # # #             self.map = self.world.get_map()

# # # #             self.traffic_manager.set_synchronous_mode(True)


# # # #     def _cleanup_actors(self):
# # # #         for actor in self.actor_handles:
# # # #             try:
# # # #                 actor.destroy()
# # # #             except RuntimeError:
# # # #                 pass
# # # #         self.actor_handles.clear()

# # # #         if self.camera_sensor:
# # # #             self.camera_sensor.destroy()
# # # #             self.camera_sensor = None

# # # #         if self.collision_sensor:
# # # #             self.collision_sensor.destroy()
# # # #             self.collision_sensor = None

# # # #         if self.lane_sensor:
# # # #             self.lane_sensor.destroy()
# # # #             self.lane_sensor = None

# # # #         if self.bev_camera:
# # # #             self.bev_camera.destroy()
# # # #             self.bev_camera = None

# # # #         self.ego = None


# # # #     def _prepare_route(self):
# # # #         assert self.world is not None
# # # #         assert self.route_config is not None

# # # #         _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
# # # #         self.route_dense = dense_route
# # # #         self.route_waypoints = [wp[0] for wp in dense_route]

# # # #         if len(self.route_waypoints) < 2:
# # # #             raise RuntimeError("The interpolated route was too short.")
        

# # # #     def _spawn_ego(self):
# # # #         assert self.world is not None
# # # #         assert self.map is not None
# # # #         assert self.route_config is not None

# # # #         bp_lib = self.world.get_blueprint_library()
# # # #         ego_bp = bp_lib.find(self.cfg.ego_filter)
# # # #         ego_bp.set_attribute("role_name", "hero")

# # # #         start_loc = self.route_config.trajectory[0]
# # # #         start_wp = self.map.get_waypoint(start_loc)
# # # #         spawn_transform = start_wp.transform
# # # #         spawn_transform.location.z += 0.5

# # # #         self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

# # # #         if self.ego is None:
# # # #             raise RuntimeError("Failed to spawn the ego vehicle at the start of the route.")
# # # #             # spawn_points = self.map.get_spawn_points()
# # # #             # self._rng.shuffle(spawn_points)
# # # #             # for sp in spawn_points:
# # # #             #     self.ego = self.world.try_spawn_actor(ego_bp, sp)
# # # #             #     if self.ego is not None:
# # # #             #         break

# # # #         if self.ego is None:
# # # #             raise RuntimeError("Failed to spawn the ego vehicle.")

# # # #         self.actor_handles.append(self.ego)

# # # #         self.collision_sensor = CollisionSensor(self.ego)
# # # #         self.lane_sensor = LaneInvasionSensor(self.ego)

# # # #         if self.cfg.render_rgb_camera:
# # # #             self.camera_sensor = CameraSensor(self.ego)

# # # #         if self.cfg.show_bev:
# # # #             self.bev_camera = BirdEyeCamera(self.ego,
# # # #                                             width=self.cfg.bev_width,
# # # #                                             height=self.cfg.bev_height,
# # # #                                             fov=self.cfg.bev_fov,
# # # #                                             z=self.cfg.bev_height_m,
# # # #                                             )
            
            
# # # #     def _warmup_ticks(self, n=10):
# # # #         assert self.world is not None
# # # #         for _ in range(n):
# # # #             self.world.tick()


# # # #     def _kmh(self, vel: carla.Vector3D) -> float:
# # # #         return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    

# # # #     def _distance(self, a: carla.Location, b: carla.Location) -> float:
# # # #         return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    

# # # #     def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
# # # #         start = max(0, self.route_index - 5)
# # # #         end = min(len(self.route_waypoints), self.route_index + window)

# # # #         best_idx = self.route_index
# # # #         best_dist = float("inf")

# # # #         for i in range(start, end):
# # # #             d = self._distance(location, self.route_waypoints[i].location)
# # # #             if d < best_dist:
# # # #                 best_dist = d
# # # #                 best_idx = i

# # # #         return best_idx
    

# # # #     def _compute_route_errors(self) -> Tuple[float, float, float, float]:
# # # #         """
# # # #         Retorna:
# # # #         lateral_error_m, heading_error_rad, progress_0_1, dist_to_goal_m
# # # #         """
# # # #         ego_tf = self.ego.get_transform()
# # # #         ego_loc = ego_tf.location
# # # #         ego_yaw = math.radians(ego_tf.rotation.yaw)

# # # #         self.route_index = self._find_nearest_route_index(ego_loc)
# # # #         wp_tf = self.route_waypoints[self.route_index]
# # # #         wp_loc = wp_tf.location
# # # #         wp_yaw = math.radians(wp_tf.rotation.yaw)

# # # #         dx = ego_loc.x - wp_loc.x
# # # #         dy = ego_loc.y - wp_loc.y

# # # #         lateral_error = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

# # # #         heading_error = ego_yaw - wp_yaw
# # # #         heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

# # # #         progress = self.route_index / max(1, (len(self.route_waypoints) - 1))
# # # #         dist_to_goal = self._distance(ego_loc, self.route_waypoints[-1].location)

# # # #         return lateral_error, heading_error, progress, dist_to_goal
    

# # # #     def _update_spectator(self):
# # # #         if not self.cfg.spectator_follow or self.ego is None or self.world is None:
# # # #             return

# # # #         spectator = self.world.get_spectator()
# # # #         ego_tf = self.ego.get_transform()
# # # #         ego_loc = ego_tf.location

# # # #         if self.cfg.spectator_rotate_with_ego:
# # # #             yaw = ego_tf.rotation.yaw
# # # #         else:
# # # #             yaw = 0.0

# # # #         spec_tf = carla.Transform(carla.Location(x=ego_loc.x, y=ego_loc.y, z=self.cfg.spectator_height_m,),
# # # #                                     carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
# # # #                                 )
        
# # # #         spectator.set_transform(spec_tf)


# # # #     def _get_obs(self) -> np.ndarray:
# # # #         vel = self.ego.get_velocity()
# # # #         acc = self.ego.get_acceleration()
# # # #         ang = self.ego.get_angular_velocity()
# # # #         ctrl = self.ego.get_control()

# # # #         speed = self._kmh(vel)
# # # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# # # #         if self.collision_sensor and len(self.collision_sensor.history) > 0:
# # # #             collided = 1.0
# # # #         else:
# # # #             collided = 0.0

        
# # # #         if self.lane_sensor:
# # # #             lane_count = self.lane_sensor.count
# # # #         else:
# # # #             lane_count = 0.0

# # # #         obs = np.array([speed / 100.0,
# # # #                         self.cfg.target_speed_kmh / 100.0,
# # # #                         lateral_error / 10.0,
# # # #                         heading_error / math.pi,
# # # #                         progress,
# # # #                         dist_to_goal / 1000.0,
# # # #                         ctrl.throttle,
# # # #                         ctrl.brake,
# # # #                         ctrl.steer,
# # # #                         # lane_count / 10.0,
# # # #                         min(lane_count, 10) / 10.0,
# # # #                         collided,
# # # #                         acc.x / 10.0,
# # # #                         acc.y / 10.0,
# # # #                         ang.z / 10.0,], dtype=np.float32,
# # # #                         )
        
# # # #         return obs
    

# # # #     def _compute_reward(self) -> float:
# # # #         speed = self._kmh(self.ego.get_velocity())
# # # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# # # #         progress_delta = progress - self.last_progress
# # # #         self.last_progress = progress

# # # #         if self.prev_dist_to_goal is None:
# # # #             dist_delta = 0.0
# # # #         else:
# # # #             dist_delta = self.prev_dist_to_goal - dist_to_goal
# # # #         self.prev_dist_to_goal = dist_to_goal

# # # #         reward = 0.0

# # # #         reward += 30.0 * progress_delta
# # # #         reward += 0.05 * dist_delta

# # # #         reward -= 0.05 * abs(lateral_error)
# # # #         reward -= 0.05 * abs(heading_error)

# # # #         speed_error = abs(speed - self.cfg.target_speed_kmh)
# # # #         reward -= 0.02 * speed_error

# # # #         if speed > 2.0:
# # # #             reward += 0.01

# # # #         if speed < 1.0:
# # # #             reward -= 0.2

# # # #         if self.collision_sensor and len(self.collision_sensor.history) > 0:
# # # #             reward -= 30.0

# # # #         if self.lane_sensor and self.lane_sensor.count > 0:
# # # #             reward -= 1.0

# # # #         if dist_to_goal < 5.0 or progress >= 0.995:
# # # #             reward += 100.0

# # # #         return reward
    

# # # #     def _check_done(self) -> Tuple[bool, bool, Dict]:
# # # #         info = {}

# # # #         speed = self._kmh(self.ego.get_velocity())
# # # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
# # # #         collided = self.collision_sensor and len(self.collision_sensor.history) > 0

# # # #         terminated = False
# # # #         truncated = False

# # # #         if speed < 1.0:
# # # #             self.stuck_steps += 1
# # # #         else:
# # # #             self.stuck_steps = 0

# # # #         if collided:
# # # #             terminated = True
# # # #             info["event"] = "collision"

# # # #         elif abs(lateral_error) > 4.0:
# # # #             terminated = True
# # # #             info["event"] = "off_route"

# # # #         elif dist_to_goal < 5.0 or progress >= 0.995:
# # # #             terminated = True
# # # #             info["event"] = "route_completed"

# # # #         elif self.stuck_steps >= 150:
# # # #             truncated = True
# # # #             info["event"] = "stuck"

# # # #         elif self.step_count >= self.cfg.max_episode_steps:
# # # #             truncated = True
# # # #             info["event"] = "time_limit"

# # # #         info["progress"] = progress
# # # #         info["dist_to_goal"] = dist_to_goal
# # # #         info["route_index"] = self.route_index
# # # #         info["speed_kmh"] = speed
# # # #         info["stuck_steps"] = self.stuck_steps
# # # #         info["lateral_error"] = lateral_error
# # # #         info["heading_error"] = heading_error

# # # #         return terminated, truncated, info
    

# # # #     def reset(self, *, seed=None, options=None):
# # # #         super().reset(seed=seed)

# # # #         if self.client is None:
# # # #             self._connect()

# # # #         if not self.route_configs:
# # # #             self._load_route_configs()

# # # #         self.step_count = 0
# # # #         self.last_progress = 0.0
# # # #         self.route_index = 0
# # # #         self.stuck_steps = 0
# # # #         self.prev_dist_to_goal = None

# # # #         if self.route_config is not None:
# # # #             old_town = self.route_config.town
# # # #         else:
# # # #             old_town = None

# # # #         self._choose_route()
# # # #         new_town = self.route_config.town

# # # #         self._cleanup_actors()

# # # #         if self.world is None or old_town != new_town:
# # # #             self._load_world_for_route()
# # # #         else:
# # # #             for _ in range(5):
# # # #                 self.world.tick()

# # # #         self._prepare_route()
# # # #         self._spawn_ego()
# # # #         self._warmup_ticks()
# # # #         self._update_spectator()

# # # #         if self.collision_sensor:
# # # #             self.collision_sensor.clear()

# # # #         if self.lane_sensor:
# # # #             self.lane_sensor.clear()

# # # #         self.last_location = self.ego.get_location()

# # # #         _, _, progress, dist_to_goal = self._compute_route_errors()
# # # #         self.last_progress = progress
# # # #         self.prev_dist_to_goal = dist_to_goal

# # # #         obs = self._get_obs()

# # # #         info = {"town": self.route_config.town,
# # # #                 "route_name": getattr(self.route_config, "name", "unknown"),
# # # #                 "route_length": len(self.route_waypoints),
# # # #                 }
                
# # # #         return obs, info
    

# # # #     def step(self, action):
# # # #         # action = np.asarray(action, dtype=np.float32)
# # # #         steer = np.clip(action[0], -1.0, 1.0)
# # # #         accel = np.clip(action[1], -1.0, 1.0)

# # # #         if accel > 0.0:
# # # #             throttle = accel
# # # #             brake = 0.0
# # # #         else:
# # # #             throttle = 0.0
# # # #             brake = -accel  

# # # #         control = carla.VehicleControl(steer=steer,
# # # #                                         throttle=throttle,
# # # #                                         brake=brake,
# # # #                                         hand_brake=False,
# # # #                                         reverse=False,
# # # #                                         manual_gear_shift=False,
# # # #                                         )
        
# # # #         self.ego.apply_control(control)
# # # #         # print(f"steer={steer:.3f}, accel={accel:.3f}, throttle={throttle:.3f}, brake={brake:.3f}, speed={self._kmh(self.ego.get_velocity()):.2f}")

# # # #         self.world.tick()
# # # #         self.step_count += 1

# # # #         self._update_spectator()

# # # #         obs = self._get_obs()
# # # #         reward = self._compute_reward()
# # # #         terminated, truncated, info = self._check_done()

# # # #         if self.cfg.show_bev and self.bev_camera is not None:
# # # #             bev = self.bev_camera.get_latest()
# # # #             if bev is not None:
# # # #                 if not self._bev_window_created:
# # # #                     cv2.namedWindow("CARLA Bird-Eye View", cv2.WINDOW_NORMAL)
# # # #                     self._bev_window_created = True

# # # #                 cv2.imshow("CARLA Bird-Eye View", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
# # # #                 cv2.waitKey(1)
# # # #                 info["bev"] = bev

# # # #         if self.cfg.render_rgb_camera and self.camera_sensor:
# # # #             rgb = self.camera_sensor.get_latest()
# # # #             if rgb is not None:
# # # #                 info["rgb"] = rgb

# # # #         return obs, reward, terminated, truncated, info
    

# # # #     # def render(self):
# # # #     #     pass


# # # #     def close(self):
# # # #         self._cleanup_actors()

# # # #         if self.world is not None and self.cfg.sync:
# # # #             settings = self.world.get_settings()
# # # #             settings.synchronous_mode = False
# # # #             settings.fixed_delta_seconds = None
# # # #             self.world.apply_settings(settings)

# # # #         if self.traffic_manager is not None and self.cfg.sync:
# # # #             try:
# # # #                 self.traffic_manager.set_synchronous_mode(False)
# # # #             except RuntimeError:
# # # #                 pass

# # # #         cv2.destroyAllWindows()
# # # ######################################################################################################################################

# # # import math
# # # import queue
# # # import random
# # # import weakref
# # # from typing import Dict, List, Optional, Tuple
# # # from types import SimpleNamespace

# # # import carla
# # # import cv2
# # # import gymnasium as gym
# # # import numpy as np
# # # from gymnasium import spaces

# # # from srunner.tools.route_parser import RouteParser
# # # from srunner.tools.route_manipulation import interpolate_trajectory
# # # from srunner.scenarios.pedestrian_crossing import PedestrianCrossing


# # # class EnvConfig:
# # #     host = "127.0.0.1"
# # #     port = 2000
# # #     traffic_manager_port = 8000
# # #     timeout = 10.0

# # #     route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
# # #     scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"
# # #     route_id = None      # All episodes will use the first route in the file if route_id is not None. Otherwise, a random route from the file will be chosen for each episode.
# # #     # route_id = "20"    # route_id can also be set to a specific value to always use the same route (useful for debugging)
# # #     # route_towns = ["Town01", "Town03", "Town05"]
# # #     route_town = "Town05"

# # #     sync = True
# # #     fixed_delta_seconds = 0.05

# # #     max_episode_steps = 500
# # #     target_speed_kmh = 30.0

# # #     ego_filter = "vehicle.lincoln.mkz_2017"
# # #     seed = 42

# # #     render_rgb_camera = True

# # #     spectator_follow = True
# # #     spectator_height_m = 40.0
# # #     spectator_rotate_with_ego = False

# # #     show_bev = True
# # #     bev_width = 800
# # #     bev_height = 800
# # #     bev_fov = 90
# # #     bev_height_m = 35.0


# # # class CollisionSensor:
# # #     def __init__(self, parent_actor: carla.Actor):
# # #         self.sensor = None
# # #         self.history = []
# # #         self._parent = parent_actor

# # #         world = self._parent.get_world()
# # #         bp = world.get_blueprint_library().find("sensor.other.collision")
# # #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# # #         weak_self = weakref.ref(self)
# # #         self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

# # #     @staticmethod
# # #     def _on_collision(weak_self, event):
# # #         self = weak_self()
# # #         if not self:
# # #             return

# # #         impulse = event.normal_impulse
# # #         intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
# # #         self.history.append((event.frame, intensity))

# # #     def clear(self):
# # #         self.history.clear()

# # #     def destroy(self):
# # #         if self.sensor is not None:
# # #             self.sensor.stop()
# # #             self.sensor.destroy()
# # #             self.sensor = None


# # # class LaneInvasionSensor:
# # #     def __init__(self, parent_actor: carla.Actor):
# # #         self.sensor = None
# # #         self.count = 0
# # #         self._parent = parent_actor

# # #         world = self._parent.get_world()
# # #         bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
# # #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# # #         weak_self = weakref.ref(self)
# # #         self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

# # #     @staticmethod
# # #     def _on_invasion(weak_self, event):
# # #         self = weak_self()
# # #         if not self:
# # #             return
# # #         self.count += 1

# # #     def clear(self):
# # #         self.count = 0

# # #     def destroy(self):
# # #         if self.sensor is not None:
# # #             self.sensor.stop()
# # #             self.sensor.destroy()
# # #             self.sensor = None


# # # class CameraSensor:
# # #     def __init__(self, parent_actor: carla.Actor, width=640, height=360):
# # #         self.sensor = None
# # #         self.queue = queue.Queue()
# # #         self.width = width
# # #         self.height = height

# # #         world = parent_actor.get_world()
# # #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# # #         bp.set_attribute("image_size_x", str(width))
# # #         bp.set_attribute("image_size_y", str(height))
# # #         bp.set_attribute("fov", "90")
# # #         bp.set_attribute("sensor_tick", "0.0")

# # #         transform = carla.Transform(carla.Location(x=1.5, z=2.4))
# # #         self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
# # #         self.sensor.listen(self.queue.put)

# # #     def get_latest(self) -> Optional[np.ndarray]:
# # #         img = None
# # #         while not self.queue.empty():
# # #             img = self.queue.get()

# # #         if img is None:
# # #             return None

# # #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# # #         array = array.reshape((self.height, self.width, 4))
# # #         array = array[:, :, :3]
# # #         return array

# # #     def destroy(self):
# # #         if self.sensor is not None:
# # #             self.sensor.stop()
# # #             self.sensor.destroy()
# # #             self.sensor = None


# # # class BirdEyeCamera:
# # #     """
# # #     Câmera top-down presa ao ego.
# # #     Como está attach_to=parent_actor com AttachmentType.Rigid,
# # #     ela acompanha a posição e a rotação do carro.
# # #     """

# # #     def __init__(self, parent_actor: carla.Actor, width=800, height=800, fov=90, z=35.0):
# # #         self.sensor = None
# # #         self.queue = queue.Queue()
# # #         self.width = width
# # #         self.height = height

# # #         world = parent_actor.get_world()
# # #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# # #         bp.set_attribute("image_size_x", str(width))
# # #         bp.set_attribute("image_size_y", str(height))
# # #         bp.set_attribute("fov", str(fov))
# # #         bp.set_attribute("sensor_tick", "0.0")

# # #         transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=z),
# # #                                     carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
# # #                                     )

# # #         self.sensor = world.spawn_actor(bp,
# # #                                         transform,
# # #                                         attach_to=parent_actor,
# # #                                         attachment_type=carla.AttachmentType.Rigid,
# # #                                         )
        
# # #         self.sensor.listen(self.queue.put)

# # #     def get_latest(self) -> Optional[np.ndarray]:
# # #         img = None
# # #         while not self.queue.empty():
# # #             img = self.queue.get()

# # #         if img is None:
# # #             return None

# # #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# # #         array = array.reshape((self.height, self.width, 4))
# # #         array = array[:, :, :3]
# # #         return array

# # #     def destroy(self):
# # #         if self.sensor is not None:
# # #             self.sensor.stop()
# # #             self.sensor.destroy()
# # #             self.sensor = None


# # # class CarlaRouteEnv(gym.Env):

# # #     def __init__(self, cfg: EnvConfig):
# # #         super().__init__()
# # #         self.cfg = cfg
# # #         self._rng = random.Random(cfg.seed)

# # #         self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
# # #                                         high=np.array([1.0, 1.0], dtype=np.float32),
# # #                                         dtype=np.float32,
# # #                                         )

# # #         self.observation_space = spaces.Box(low=-np.inf,
# # #                                             high=np.inf,
# # #                                             shape=(14,),
# # #                                             dtype=np.float32,
# # #                                             )

# # #         self.client: Optional[carla.Client] = None
# # #         self.world: Optional[carla.World] = None
# # #         self.map: Optional[carla.Map] = None
# # #         self.traffic_manager = None

# # #         self.route_configs = []
# # #         self.route_config = None
# # #         self.route_dense: List[Tuple[carla.Transform, object]] = []
# # #         self.route_waypoints: List[carla.Transform] = []
# # #         self.route_index = 0

# # #         self.ego: Optional[carla.Vehicle] = None
# # #         self.collision_sensor: Optional[CollisionSensor] = None
# # #         self.lane_sensor: Optional[LaneInvasionSensor] = None
# # #         self.camera_sensor: Optional[CameraSensor] = None
# # #         self.bev_camera: Optional[BirdEyeCamera] = None

# # #         self.actor_handles: List[carla.Actor] = []

# # #         self.step_count = 0
# # #         self.last_progress = 0.0
# # #         self.last_location: Optional[carla.Location] = None
# # #         self.stuck_steps = 0
# # #         self.prev_dist_to_goal = None

# # #         self._bev_window_created = False


# # #     def _connect(self):
# # #         self.client = carla.Client(self.cfg.host, self.cfg.port)
# # #         self.client.set_timeout(self.cfg.timeout)
# # #         self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)
# # #         # self.traffic_manager = None


# # #     def _load_route_configs(self):
# # #         self.route_configs = RouteParser.parse_routes_file(self.cfg.route_file,
# # #                                                             self.cfg.scenario_file,
# # #                                                             single_route=self.cfg.route_id,
# # #                                                             )
        
# # #         if not self.route_configs:
# # #             raise RuntimeError("No route found in the XML file.")
        

# # #     def _choose_route(self):
# # #         if self.cfg.route_id is not None:
# # #             self.route_config = self.route_configs[0]
# # #             # print('====== ANA ======', self.route_config.town)

# # #         route_town = getattr(self.cfg, "route_town", None)

# # #         if route_town is not None:
# # #             valid_routes = []
# # #             for rc in self.route_configs:
# # #                 if rc.town == self.cfg.route_town:
# # #                     valid_routes.append(rc)

# # #             if not valid_routes: 
# # #                 raise RuntimeError(f"No route found for town {self.cfg.route_town}") 

# # #             self.route_config = self._rng.choice(valid_routes)  

# # #         else:
# # #             self.route_config = self._rng.choice(self.route_configs)
                


# # #     def _load_world_for_route(self):
# # #         assert self.client is not None
# # #         assert self.route_config is not None

# # #         town = self.route_config.town
# # #         self.world = self.client.load_world(town)
# # #         self.map = self.world.get_map()

# # #         if self.cfg.sync:
# # #             settings = self.world.get_settings()
# # #             settings.synchronous_mode = True
# # #             settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
# # #             self.world.apply_settings(settings)

# # #             self.client.reload_world(False)
# # #             self.world = self.client.get_world()
# # #             self.map = self.world.get_map()

# # #             self.traffic_manager.set_synchronous_mode(True)


# # #     def _cleanup_actors(self):
# # #         for actor in self.actor_handles:
# # #             try:
# # #                 actor.destroy()
# # #             except RuntimeError:
# # #                 pass
# # #         self.actor_handles.clear()

# # #         if self.camera_sensor:
# # #             self.camera_sensor.destroy()
# # #             self.camera_sensor = None

# # #         if self.collision_sensor:
# # #             self.collision_sensor.destroy()
# # #             self.collision_sensor = None

# # #         if self.lane_sensor:
# # #             self.lane_sensor.destroy()
# # #             self.lane_sensor = None

# # #         if self.bev_camera:
# # #             self.bev_camera.destroy()
# # #             self.bev_camera = None

# # #         self.ego = None


# # #     def _prepare_route(self):
# # #         assert self.world is not None
# # #         assert self.route_config is not None

# # #         _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
# # #         self.route_dense = dense_route
# # #         self.route_waypoints = [wp[0] for wp in dense_route]

# # #         if len(self.route_waypoints) < 2:
# # #             raise RuntimeError("The interpolated route was too short.")
        

# # #     def _spawn_ego(self):
# # #         assert self.world is not None
# # #         assert self.map is not None
# # #         assert self.route_config is not None

# # #         bp_lib = self.world.get_blueprint_library()
# # #         ego_bp = bp_lib.find(self.cfg.ego_filter)
# # #         ego_bp.set_attribute("role_name", "hero")

# # #         start_loc = self.route_config.trajectory[0]
# # #         start_wp = self.map.get_waypoint(start_loc)
# # #         spawn_transform = start_wp.transform
# # #         spawn_transform.location.z += 0.5

# # #         self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

# # #         if self.ego is None:
# # #             raise RuntimeError("Failed to spawn the ego vehicle at the start of the route.")
# # #             # spawn_points = self.map.get_spawn_points()
# # #             # self._rng.shuffle(spawn_points)
# # #             # for sp in spawn_points:
# # #             #     self.ego = self.world.try_spawn_actor(ego_bp, sp)
# # #             #     if self.ego is not None:
# # #             #         break

# # #         if self.ego is None:
# # #             raise RuntimeError("Failed to spawn the ego vehicle.")

# # #         self.actor_handles.append(self.ego)

# # #         self.collision_sensor = CollisionSensor(self.ego)
# # #         self.lane_sensor = LaneInvasionSensor(self.ego)

# # #         if self.cfg.render_rgb_camera:
# # #             self.camera_sensor = CameraSensor(self.ego)

# # #         if self.cfg.show_bev:
# # #             self.bev_camera = BirdEyeCamera(self.ego,
# # #                                             width=self.cfg.bev_width,
# # #                                             height=self.cfg.bev_height,
# # #                                             fov=self.cfg.bev_fov,
# # #                                             z=self.cfg.bev_height_m,
# # #                                             )
            
            
# # #     def _warmup_ticks(self, n=10):
# # #         assert self.world is not None
# # #         for _ in range(n):
# # #             self.world.tick()


# # #     def _kmh(self, vel: carla.Vector3D) -> float:
# # #         return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    

# # #     def _distance(self, a: carla.Location, b: carla.Location) -> float:
# # #         return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    

# # #     def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
# # #         start = max(0, self.route_index - 5)
# # #         end = min(len(self.route_waypoints), self.route_index + window)

# # #         best_idx = self.route_index
# # #         best_dist = float("inf")

# # #         for i in range(start, end):
# # #             d = self._distance(location, self.route_waypoints[i].location)
# # #             if d < best_dist:
# # #                 best_dist = d
# # #                 best_idx = i

# # #         return best_idx
    

# # #     def _compute_route_errors(self) -> Tuple[float, float, float, float]:
# # #         """
# # #         Retorna:
# # #         lateral_error_m, heading_error_rad, progress_0_1, dist_to_goal_m
# # #         """
# # #         ego_tf = self.ego.get_transform()
# # #         ego_loc = ego_tf.location
# # #         ego_yaw = math.radians(ego_tf.rotation.yaw)

# # #         self.route_index = self._find_nearest_route_index(ego_loc)
# # #         wp_tf = self.route_waypoints[self.route_index]
# # #         wp_loc = wp_tf.location
# # #         wp_yaw = math.radians(wp_tf.rotation.yaw)

# # #         dx = ego_loc.x - wp_loc.x
# # #         dy = ego_loc.y - wp_loc.y

# # #         lateral_error = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

# # #         heading_error = ego_yaw - wp_yaw
# # #         heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

# # #         progress = self.route_index / max(1, (len(self.route_waypoints) - 1))
# # #         dist_to_goal = self._distance(ego_loc, self.route_waypoints[-1].location)

# # #         return lateral_error, heading_error, progress, dist_to_goal
    

# # #     def _update_spectator(self):
# # #         if not self.cfg.spectator_follow or self.ego is None or self.world is None:
# # #             return

# # #         spectator = self.world.get_spectator()
# # #         ego_tf = self.ego.get_transform()
# # #         ego_loc = ego_tf.location

# # #         if self.cfg.spectator_rotate_with_ego:
# # #             yaw = ego_tf.rotation.yaw
# # #         else:
# # #             yaw = 0.0

# # #         spec_tf = carla.Transform(carla.Location(x=ego_loc.x, y=ego_loc.y, z=self.cfg.spectator_height_m,),
# # #                                     carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
# # #                                 )
        
# # #         spectator.set_transform(spec_tf)


# # #     def _get_obs(self) -> np.ndarray:
# # #         vel = self.ego.get_velocity()
# # #         acc = self.ego.get_acceleration()
# # #         ang = self.ego.get_angular_velocity()
# # #         ctrl = self.ego.get_control()

# # #         speed = self._kmh(vel)
# # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# # #         if self.collision_sensor and len(self.collision_sensor.history) > 0:
# # #             collided = 1.0
# # #         else:
# # #             collided = 0.0

        
# # #         if self.lane_sensor:
# # #             lane_count = self.lane_sensor.count
# # #         else:
# # #             lane_count = 0.0

# # #         obs = np.array([speed / 100.0,
# # #                         self.cfg.target_speed_kmh / 100.0,
# # #                         lateral_error / 10.0,
# # #                         heading_error / math.pi,
# # #                         progress,
# # #                         dist_to_goal / 1000.0,
# # #                         ctrl.throttle,
# # #                         ctrl.brake,
# # #                         ctrl.steer,
# # #                         # lane_count / 10.0,
# # #                         min(lane_count, 10) / 10.0,
# # #                         collided,
# # #                         acc.x / 10.0,
# # #                         acc.y / 10.0,
# # #                         ang.z / 10.0,], dtype=np.float32,
# # #                         )
        
# # #         return obs
    

# # #     def _compute_reward(self) -> float:
# # #         speed = self._kmh(self.ego.get_velocity())
# # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# # #         progress_delta = progress - self.last_progress
# # #         self.last_progress = progress

# # #         if self.prev_dist_to_goal is None:
# # #             dist_delta = 0.0
# # #         else:
# # #             dist_delta = self.prev_dist_to_goal - dist_to_goal
# # #         self.prev_dist_to_goal = dist_to_goal

# # #         reward = 0.0

# # #         reward += 30.0 * progress_delta
# # #         reward += 0.05 * dist_delta

# # #         reward -= 0.05 * abs(lateral_error)
# # #         reward -= 0.05 * abs(heading_error)

# # #         speed_error = abs(speed - self.cfg.target_speed_kmh)
# # #         reward -= 0.02 * speed_error

# # #         if speed > 2.0:
# # #             reward += 0.01

# # #         if speed < 1.0:
# # #             reward -= 0.2

# # #         if self.collision_sensor and len(self.collision_sensor.history) > 0:
# # #             reward -= 30.0

# # #         if self.lane_sensor and self.lane_sensor.count > 0:
# # #             reward -= 1.0

# # #         if dist_to_goal < 5.0 or progress >= 0.995:
# # #             reward += 100.0

# # #         return reward
    

# # #     def _check_done(self) -> Tuple[bool, bool, Dict]:
# # #         info = {}

# # #         speed = self._kmh(self.ego.get_velocity())
# # #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
# # #         collided = self.collision_sensor and len(self.collision_sensor.history) > 0

# # #         terminated = False
# # #         truncated = False

# # #         if speed < 1.0:
# # #             self.stuck_steps += 1
# # #         else:
# # #             self.stuck_steps = 0

# # #         if collided:
# # #             terminated = True
# # #             info["event"] = "collision"

# # #         elif abs(lateral_error) > 4.0:
# # #             terminated = True
# # #             info["event"] = "off_route"

# # #         elif dist_to_goal < 5.0 or progress >= 0.995:
# # #             terminated = True
# # #             info["event"] = "route_completed"

# # #         elif self.stuck_steps >= 150:
# # #             truncated = True
# # #             info["event"] = "stuck"

# # #         elif self.step_count >= self.cfg.max_episode_steps:
# # #             truncated = True
# # #             info["event"] = "time_limit"

# # #         info["progress"] = progress
# # #         info["dist_to_goal"] = dist_to_goal
# # #         info["route_index"] = self.route_index
# # #         info["speed_kmh"] = speed
# # #         info["stuck_steps"] = self.stuck_steps
# # #         info["lateral_error"] = lateral_error
# # #         info["heading_error"] = heading_error

# # #         return terminated, truncated, info
    

# # #     # def _build_pedestrian_config(self):
# # #     #     # Escolhe um ponto da rota onde o cenário vai acontecer.
# # #     #     # Pode começar com um waypoint fixo da rota, por exemplo um do meio.
# # #     #     trigger_idx = max(5, len(self.route_waypoints) // 3)
# # #     #     trigger_tf = self.route_waypoints[trigger_idx]

# # #     #     config = SimpleNamespace()
# # #     #     config.trigger_points = [trigger_tf]
# # #     #     config.route = [(wp, None) for wp in self.route_waypoints]

# # #     #     # campos usados por BasicScenario
# # #     #     config.weather = carla.WeatherParameters(sun_altitude_angle=70.0)
# # #     #     config.friction = None
# # #     #     config.other_actors = []
# # #     #     config.route_var_name = None

# # #     #     return config

# # #     def _build_pedestrian_config(self):
# # #         if not self.route_waypoints:
# # #             raise RuntimeError(
# # #                 "self.route_waypoints está vazio. "
# # #                 "Chame _prepare_route() antes de _build_pedestrian_config()."
# # #             )

# # #         trigger_idx = max(5, len(self.route_waypoints) // 3)
# # #         trigger_idx = min(trigger_idx, len(self.route_waypoints) - 1)

# # #         trigger_tf = self.route_waypoints[trigger_idx]

# # #         from types import SimpleNamespace
# # #         import carla

# # #         config = SimpleNamespace()
# # #         config.trigger_points = [trigger_tf]
# # #         config.route = [(wp, None) for wp in self.route_waypoints]
# # #         config.weather = carla.WeatherParameters(sun_altitude_angle=70.0)
# # #         config.friction = None
# # #         config.other_actors = []
# # #         config.route_var_name = None

# # #         return config
    

# # #     # def reset(self, *, seed=None, options=None):
# # #     #     super().reset(seed=seed)

# # #     #     if self.client is None:
# # #     #         self._connect()

# # #     #     if not self.route_configs:
# # #     #         self._load_route_configs()

  
# # #     #     self.active_scenario = PedestrianCrossing(world=self.world,
# # #     #                                             ego_vehicles=[self.ego],
# # #     #                                             config=self.scenario_config,
# # #     #                                             debug_mode=False,
# # #     #                                             criteria_enable=False,
# # #     #                                             timeout=60,
# # #     #                                             ) 

# # #     #     self.step_count = 0
# # #     #     self.last_progress = 0.0
# # #     #     self.route_index = 0
# # #     #     self.stuck_steps = 0
# # #     #     self.prev_dist_to_goal = None

# # #     #     if self.route_config is not None:
# # #     #         old_town = self.route_config.town
# # #     #     else:
# # #     #         old_town = None

# # #     #     self._choose_route()
# # #     #     new_town = self.route_config.town

# # #     #     self._cleanup_actors()

# # #     #     if self.world is None or old_town != new_town:
# # #     #         self._load_world_for_route()
# # #     #     else:
# # #     #         for _ in range(5):
# # #     #             self.world.tick()

# # #     #     self._prepare_route()
# # #     #     self._spawn_ego()
# # #     #     self._warmup_ticks()
# # #     #     self.scenario_config = self._build_pedestrian_config() 
# # #     #     self._build_pedestrian_scenario()
# # #     #     self._update_spectator()

# # #     #     if self.collision_sensor:
# # #     #         self.collision_sensor.clear()

# # #     #     if self.lane_sensor:
# # #     #         self.lane_sensor.clear()

# # #     #     self.last_location = self.ego.get_location()

# # #     #     _, _, progress, dist_to_goal = self._compute_route_errors()
# # #     #     self.last_progress = progress
# # #     #     self.prev_dist_to_goal = dist_to_goal

# # #     #     obs = self._get_obs()

# # #     #     info = {"town": self.route_config.town,
# # #     #             "route_name": getattr(self.route_config, "name", "unknown"),
# # #     #             "route_length": len(self.route_waypoints),
# # #     #             }
                
# # #     #     return obs, info

# # #     def reset(self, *, seed=None, options=None):
# # #         super().reset(seed=seed)

# # #         if self.client is None:
# # #             self._connect()

# # #         if not self.route_configs:
# # #             self._load_route_configs()

# # #         self.step_count = 0
# # #         self.last_progress = 0.0
# # #         self.route_index = 0
# # #         self.stuck_steps = 0
# # #         self.prev_dist_to_goal = None

# # #         old_town = self.route_config.town if self.route_config is not None else None

# # #         self._choose_route()
# # #         new_town = self.route_config.town

# # #         self._cleanup_actors()

# # #         if self.world is None or old_town != new_town:
# # #             self._load_world_for_route()
# # #         else:
# # #             for _ in range(5):
# # #                 self.world.tick()

# # #         self._prepare_route()
# # #         self._spawn_ego()
# # #         self._warmup_ticks()
# # #         self._update_spectator()

# # #         # só agora a rota já existe
# # #         self.scenario_config = self._build_pedestrian_config()
# # #         self._build_pedestrian_scenario()

# # #         if self.collision_sensor:
# # #             self.collision_sensor.clear()
# # #         if self.lane_sensor:
# # #             self.lane_sensor.clear()

# # #         self.last_location = self.ego.get_location()

# # #         _, _, progress, dist_to_goal = self._compute_route_errors()
# # #         self.last_progress = progress
# # #         self.prev_dist_to_goal = dist_to_goal

# # #         obs = self._get_obs()
# # #         info = {
# # #             "town": self.route_config.town,
# # #             "route_name": getattr(self.route_config, "name", "unknown"),
# # #             "route_length": len(self.route_waypoints),
# # #         }
# # #         return obs, info
    

# # #     def step(self, action):
# # #         # action = np.asarray(action, dtype=np.float32)
# # #         steer = np.clip(action[0], -1.0, 1.0)
# # #         accel = np.clip(action[1], -1.0, 1.0)

# # #         if accel > 0.0:
# # #             throttle = accel
# # #             brake = 0.0
# # #         else:
# # #             throttle = 0.0
# # #             brake = -accel  

# # #         control = carla.VehicleControl(steer=steer,
# # #                                         throttle=throttle,
# # #                                         brake=brake,
# # #                                         hand_brake=False,
# # #                                         reverse=False,
# # #                                         manual_gear_shift=False,
# # #                                         )
        
# # #         self.ego.apply_control(control)
# # #         # print(f"steer={steer:.3f}, accel={accel:.3f}, throttle={throttle:.3f}, brake={brake:.3f}, speed={self._kmh(self.ego.get_velocity()):.2f}")

# # #         self.world.tick()
# # #         self.step_count += 1

# # #         self._update_spectator()


# # #         if self.active_scenario is not None:
# # #             self.active_scenario.scenario.scenario_tree.tick_once()

# # #         obs = self._get_obs()
# # #         reward = self._compute_reward()
# # #         terminated, truncated, info = self._check_done()

# # #         if self.cfg.show_bev and self.bev_camera is not None:
# # #             bev = self.bev_camera.get_latest()
# # #             if bev is not None:
# # #                 if not self._bev_window_created:
# # #                     cv2.namedWindow("CARLA Bird-Eye View", cv2.WINDOW_NORMAL)
# # #                     self._bev_window_created = True

# # #                 cv2.imshow("CARLA Bird-Eye View", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
# # #                 cv2.waitKey(1)
# # #                 info["bev"] = bev

# # #         if self.cfg.render_rgb_camera and self.camera_sensor:
# # #             rgb = self.camera_sensor.get_latest()
# # #             if rgb is not None:
# # #                 info["rgb"] = rgb

# # #         return obs, reward, terminated, truncated, info
    

# # #     # def render(self):
# # #     #     pass


# # #     def close(self):
# # #         self._cleanup_actors()

# # #         if self.world is not None and self.cfg.sync:
# # #             settings = self.world.get_settings()
# # #             settings.synchronous_mode = False
# # #             settings.fixed_delta_seconds = None
# # #             self.world.apply_settings(settings)

# # #         if getattr(self, "active_scenario", None) is not None:
# # #             try:
# # #                 self.active_scenario.remove_all_actors()
# # #             except Exception:
# # #                 pass
# # #             self.active_scenario = None

# # #         if self.traffic_manager is not None and self.cfg.sync:
# # #             try:
# # #                 self.traffic_manager.set_synchronous_mode(False)
# # #             except RuntimeError:
# # #                 pass

# # #         cv2.destroyAllWindows()
# # ##########################################################################################################


# # import math
# # import queue
# # import random
# # import weakref
# # from types import SimpleNamespace
# # from typing import Dict, List, Optional, Tuple

# # import carla
# # import cv2
# # import gymnasium as gym
# # import numpy as np
# # from gymnasium import spaces

# # from srunner.tools.route_parser import RouteParser
# # from srunner.tools.route_manipulation import interpolate_trajectory
# # from srunner.scenarios.pedestrian_crossing import PedestrianCrossing



# # class EnvConfig:
# #     host = "127.0.0.1"
# #     port = 2000
# #     traffic_manager_port = 8000
# #     timeout = 10.0

# #     route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
# #     scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"

# #     # Se route_id != None, usa uma rota específica do XML
# #     route_id = None
# #     # Exemplo:
# #     # route_id = "20"

# #     # Se route_town != None e route_id == None, escolhe aleatoriamente uma rota dessa town
# #     route_town = "Town05"

# #     sync = True
# #     fixed_delta_seconds = 0.05

# #     max_episode_steps = 500
# #     target_speed_kmh = 30.0

# #     ego_filter = "vehicle.lincoln.mkz_2017"
# #     seed = 42

# #     # Sensor frontal RGB opcional
# #     render_rgb_camera = False

# #     # Spectator da janela do CARLA
# #     spectator_follow = True
# #     spectator_height_m = 40.0
# #     spectator_rotate_with_ego = False

# #     # Câmera BEV opcional
# #     show_bev = False
# #     bev_width = 800
# #     bev_height = 800
# #     bev_fov = 90
# #     bev_height_m = 35.0


# # class CollisionSensor:
# #     def __init__(self, parent_actor: carla.Actor):
# #         self.sensor = None
# #         self.history = []
# #         self._parent = parent_actor

# #         world = self._parent.get_world()
# #         bp = world.get_blueprint_library().find("sensor.other.collision")
# #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# #         weak_self = weakref.ref(self)
# #         self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

# #     @staticmethod
# #     def _on_collision(weak_self, event):
# #         self = weak_self()
# #         if not self:
# #             return

# #         impulse = event.normal_impulse
# #         intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
# #         self.history.append((event.frame, intensity))

# #     def clear(self):
# #         self.history.clear()

# #     def destroy(self):
# #         if self.sensor is not None:
# #             self.sensor.stop()
# #             self.sensor.destroy()
# #             self.sensor = None


# # class LaneInvasionSensor:
# #     def __init__(self, parent_actor: carla.Actor):
# #         self.sensor = None
# #         self.count = 0
# #         self._parent = parent_actor

# #         world = self._parent.get_world()
# #         bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
# #         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

# #         weak_self = weakref.ref(self)
# #         self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

# #     @staticmethod
# #     def _on_invasion(weak_self, event):
# #         self = weak_self()
# #         if not self:
# #             return
# #         self.count += 1

# #     def clear(self):
# #         self.count = 0

# #     def destroy(self):
# #         if self.sensor is not None:
# #             self.sensor.stop()
# #             self.sensor.destroy()
# #             self.sensor = None


# # class CameraSensor:
# #     def __init__(self, parent_actor: carla.Actor, width=640, height=360):
# #         self.sensor = None
# #         self.queue = queue.Queue()
# #         self.width = width
# #         self.height = height

# #         world = parent_actor.get_world()
# #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# #         bp.set_attribute("image_size_x", str(width))
# #         bp.set_attribute("image_size_y", str(height))
# #         bp.set_attribute("fov", "90")
# #         bp.set_attribute("sensor_tick", "0.0")

# #         transform = carla.Transform(carla.Location(x=1.5, z=2.4))
# #         self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
# #         self.sensor.listen(self.queue.put)

# #     def get_latest(self) -> Optional[np.ndarray]:
# #         img = None
# #         while not self.queue.empty():
# #             img = self.queue.get()

# #         if img is None:
# #             return None

# #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# #         array = array.reshape((self.height, self.width, 4))
# #         array = array[:, :, :3]
# #         return array

# #     def destroy(self):
# #         if self.sensor is not None:
# #             self.sensor.stop()
# #             self.sensor.destroy()
# #             self.sensor = None


# # class BirdEyeCamera:
# #     """
# #     Câmera top-down presa ao ego.
# #     Como está attach_to=parent_actor com AttachmentType.Rigid,
# #     ela acompanha a posição e a rotação do carro.
# #     """

# #     def __init__(self, parent_actor: carla.Actor, width=800, height=800, fov=90, z=35.0):
# #         self.sensor = None
# #         self.queue = queue.Queue()
# #         self.width = width
# #         self.height = height

# #         world = parent_actor.get_world()
# #         bp = world.get_blueprint_library().find("sensor.camera.rgb")
# #         bp.set_attribute("image_size_x", str(width))
# #         bp.set_attribute("image_size_y", str(height))
# #         bp.set_attribute("fov", str(fov))
# #         bp.set_attribute("sensor_tick", "0.0")

# #         transform = carla.Transform(
# #             carla.Location(x=0.0, y=0.0, z=z),
# #             carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
# #         )

# #         self.sensor = world.spawn_actor(
# #             bp,
# #             transform,
# #             attach_to=parent_actor,
# #             attachment_type=carla.AttachmentType.Rigid,
# #         )
# #         self.sensor.listen(self.queue.put)

# #     def get_latest(self) -> Optional[np.ndarray]:
# #         img = None
# #         while not self.queue.empty():
# #             img = self.queue.get()

# #         if img is None:
# #             return None

# #         array = np.frombuffer(img.raw_data, dtype=np.uint8)
# #         array = array.reshape((self.height, self.width, 4))
# #         array = array[:, :, :3]
# #         return array

# #     def destroy(self):
# #         if self.sensor is not None:
# #             self.sensor.stop()
# #             self.sensor.destroy()
# #             self.sensor = None


# # class CarlaRouteEnv(gym.Env):
# #     def __init__(self, cfg: EnvConfig):
# #         super().__init__()
# #         self.cfg = cfg
# #         self._rng = random.Random(cfg.seed)

# #         # action[0] = steer em [-1, 1]
# #         # action[1] = accel em [-1, 1]
# #         #   accel > 0 -> throttle
# #         #   accel < 0 -> brake
# #         self.action_space = spaces.Box(
# #             low=np.array([-1.0, -1.0], dtype=np.float32),
# #             high=np.array([1.0, 1.0], dtype=np.float32),
# #             dtype=np.float32,
# #         )

# #         # [speed, target_speed, lateral_error, heading_error,
# #         #  progress, dist_to_goal, throttle, brake, steer,
# #         #  lane_count, collided, accel_x, accel_y, ang_vel_z]
# #         self.observation_space = spaces.Box(
# #             low=-np.inf,
# #             high=np.inf,
# #             shape=(14,),
# #             dtype=np.float32,
# #         )

# #         self.client: Optional[carla.Client] = None
# #         self.world: Optional[carla.World] = None
# #         self.map: Optional[carla.Map] = None
# #         self.traffic_manager = None

# #         self.route_configs = []
# #         self.route_config = None
# #         self.route_dense: List[Tuple[carla.Transform, object]] = []
# #         self.route_waypoints: List[carla.Transform] = []
# #         self.route_index = 0

# #         self.ego: Optional[carla.Vehicle] = None
# #         self.collision_sensor: Optional[CollisionSensor] = None
# #         self.lane_sensor: Optional[LaneInvasionSensor] = None
# #         self.camera_sensor: Optional[CameraSensor] = None
# #         self.bev_camera: Optional[BirdEyeCamera] = None

# #         self.actor_handles: List[carla.Actor] = []

# #         self.step_count = 0
# #         self.last_progress = 0.0
# #         self.last_location: Optional[carla.Location] = None
# #         self.stuck_steps = 0
# #         self.prev_dist_to_goal = None

# #         self.active_scenario = None
# #         self.scenario_config = None

# #         self._bev_window_created = False

# #     def _connect(self):
# #         self.client = carla.Client(self.cfg.host, self.cfg.port)
# #         self.client.set_timeout(self.cfg.timeout)
# #         self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)

# #     def _load_route_configs(self):
# #         self.route_configs = RouteParser.parse_routes_file(
# #             self.cfg.route_file,
# #             self.cfg.scenario_file,
# #             single_route=self.cfg.route_id,
# #         )

# #         if not self.route_configs:
# #             raise RuntimeError("No route found in the XML file.")

# #     def _choose_route(self):
# #         # Se route_id está definido, parse_routes_file já filtrou.
# #         if self.cfg.route_id is not None:
# #             self.route_config = self.route_configs[0]
# #             return

# #         route_town = getattr(self.cfg, "route_town", None)

# #         if route_town is not None:
# #             valid_routes = [rc for rc in self.route_configs if rc.town == route_town]

# #             if not valid_routes:
# #                 raise RuntimeError(f"No route found for town {route_town}")

# #             self.route_config = self._rng.choice(valid_routes)
# #         else:
# #             self.route_config = self._rng.choice(self.route_configs)

# #     def _load_world_for_route(self):
# #         assert self.client is not None
# #         assert self.route_config is not None

# #         town = self.route_config.town
# #         self.world = self.client.load_world(town)
# #         self.map = self.world.get_map()

# #         if self.cfg.sync:
# #             settings = self.world.get_settings()
# #             settings.synchronous_mode = True
# #             settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
# #             self.world.apply_settings(settings)

# #             self.client.reload_world(False)
# #             self.world = self.client.get_world()
# #             self.map = self.world.get_map()

# #             if self.traffic_manager is not None:
# #                 self.traffic_manager.set_synchronous_mode(True)

# #     def _cleanup_actors(self):
# #         # primeiro remove atores do cenário customizado
# #         if self.active_scenario is not None:
# #             try:
# #                 self.active_scenario.remove_all_actors()
# #             except Exception:
# #                 pass
# #             self.active_scenario = None
# #             self.scenario_config = None

# #         if self.camera_sensor:
# #             self.camera_sensor.destroy()
# #             self.camera_sensor = None

# #         if self.collision_sensor:
# #             self.collision_sensor.destroy()
# #             self.collision_sensor = None

# #         if self.lane_sensor:
# #             self.lane_sensor.destroy()
# #             self.lane_sensor = None

# #         if self.bev_camera:
# #             self.bev_camera.destroy()
# #             self.bev_camera = None

# #         for actor in self.actor_handles:
# #             try:
# #                 actor.destroy()
# #             except RuntimeError:
# #                 pass
# #         self.actor_handles.clear()

# #         self.ego = None

# #     def _prepare_route(self):
# #         assert self.world is not None
# #         assert self.route_config is not None

# #         _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
# #         self.route_dense = dense_route
# #         self.route_waypoints = [wp[0] for wp in dense_route]

# #         if len(self.route_waypoints) < 2:
# #             raise RuntimeError("The interpolated route was too short.")

# #     def _spawn_ego(self):
# #         assert self.world is not None
# #         assert self.map is not None
# #         assert self.route_config is not None

# #         bp_lib = self.world.get_blueprint_library()
# #         ego_bp = bp_lib.find(self.cfg.ego_filter)
# #         ego_bp.set_attribute("role_name", "hero")

# #         start_loc = self.route_config.trajectory[0]
# #         start_wp = self.map.get_waypoint(start_loc)
# #         spawn_transform = start_wp.transform
# #         spawn_transform.location.z += 0.5

# #         self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

# #         if self.ego is None:
# #             # fallback para outro spawn point
# #             spawn_points = self.map.get_spawn_points()
# #             self._rng.shuffle(spawn_points)
# #             for sp in spawn_points:
# #                 self.ego = self.world.try_spawn_actor(ego_bp, sp)
# #                 if self.ego is not None:
# #                     break

# #         if self.ego is None:
# #             raise RuntimeError("Failed to spawn the ego vehicle.")

# #         self.actor_handles.append(self.ego)

# #         self.collision_sensor = CollisionSensor(self.ego)
# #         self.lane_sensor = LaneInvasionSensor(self.ego)

# #         if self.cfg.render_rgb_camera:
# #             self.camera_sensor = CameraSensor(self.ego)

# #         if self.cfg.show_bev:
# #             self.bev_camera = BirdEyeCamera(
# #                 self.ego,
# #                 width=self.cfg.bev_width,
# #                 height=self.cfg.bev_height,
# #                 fov=self.cfg.bev_fov,
# #                 z=self.cfg.bev_height_m,
# #             )

# #     def _warmup_ticks(self, n=10):
# #         assert self.world is not None
# #         for _ in range(n):
# #             self.world.tick()

# #     def _kmh(self, vel: carla.Vector3D) -> float:
# #         return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

# #     def _distance(self, a: carla.Location, b: carla.Location) -> float:
# #         return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

# #     def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
# #         start = max(0, self.route_index - 5)
# #         end = min(len(self.route_waypoints), self.route_index + window)

# #         best_idx = self.route_index
# #         best_dist = float("inf")

# #         for i in range(start, end):
# #             d = self._distance(location, self.route_waypoints[i].location)
# #             if d < best_dist:
# #                 best_dist = d
# #                 best_idx = i

# #         return best_idx

# #     def _compute_route_errors(self) -> Tuple[float, float, float, float]:
# #         """
# #         Retorna:
# #         lateral_error_m, heading_error_rad, progress_0_1, dist_to_goal_m
# #         """
# #         assert self.ego is not None
# #         assert len(self.route_waypoints) > 0

# #         ego_tf = self.ego.get_transform()
# #         ego_loc = ego_tf.location
# #         ego_yaw = math.radians(ego_tf.rotation.yaw)

# #         self.route_index = self._find_nearest_route_index(ego_loc)
# #         wp_tf = self.route_waypoints[self.route_index]
# #         wp_loc = wp_tf.location
# #         wp_yaw = math.radians(wp_tf.rotation.yaw)

# #         dx = ego_loc.x - wp_loc.x
# #         dy = ego_loc.y - wp_loc.y

# #         lateral_error = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

# #         heading_error = ego_yaw - wp_yaw
# #         heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

# #         progress = self.route_index / max(1, (len(self.route_waypoints) - 1))
# #         dist_to_goal = self._distance(ego_loc, self.route_waypoints[-1].location)

# #         return lateral_error, heading_error, progress, dist_to_goal

# #     def _update_spectator(self):
# #         if not self.cfg.spectator_follow or self.ego is None or self.world is None:
# #             return

# #         spectator = self.world.get_spectator()
# #         ego_tf = self.ego.get_transform()
# #         ego_loc = ego_tf.location

# #         yaw = ego_tf.rotation.yaw if self.cfg.spectator_rotate_with_ego else 0.0

# #         spec_tf = carla.Transform(
# #             carla.Location(x=ego_loc.x, y=ego_loc.y, z=self.cfg.spectator_height_m),
# #             carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
# #         )
# #         spectator.set_transform(spec_tf)

# #     def _get_obs(self) -> np.ndarray:
# #         assert self.ego is not None

# #         vel = self.ego.get_velocity()
# #         acc = self.ego.get_acceleration()
# #         ang = self.ego.get_angular_velocity()
# #         ctrl = self.ego.get_control()

# #         speed = self._kmh(vel)
# #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# #         collided = 1.0 if (self.collision_sensor and len(self.collision_sensor.history) > 0) else 0.0
# #         lane_count = float(self.lane_sensor.count if self.lane_sensor else 0.0)

# #         obs = np.array(
# #             [
# #                 speed / 100.0,
# #                 self.cfg.target_speed_kmh / 100.0,
# #                 lateral_error / 10.0,
# #                 heading_error / math.pi,
# #                 progress,
# #                 dist_to_goal / 1000.0,
# #                 float(ctrl.throttle),
# #                 float(ctrl.brake),
# #                 float(ctrl.steer),
# #                 min(lane_count, 10.0) / 10.0,
# #                 collided,
# #                 float(acc.x) / 10.0,
# #                 float(acc.y) / 10.0,
# #                 float(ang.z) / 10.0,
# #             ],
# #             dtype=np.float32,
# #         )
# #         return obs

# #     def _compute_reward(self) -> float:
# #         assert self.ego is not None

# #         speed = self._kmh(self.ego.get_velocity())
# #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

# #         progress_delta = progress - self.last_progress
# #         self.last_progress = progress

# #         if self.prev_dist_to_goal is None:
# #             dist_delta = 0.0
# #         else:
# #             dist_delta = self.prev_dist_to_goal - dist_to_goal
# #         self.prev_dist_to_goal = dist_to_goal

# #         reward = 0.0

# #         reward += 30.0 * progress_delta
# #         reward += 0.05 * dist_delta

# #         reward -= 0.05 * abs(lateral_error)
# #         reward -= 0.05 * abs(heading_error)

# #         speed_error = abs(speed - self.cfg.target_speed_kmh)
# #         reward -= 0.02 * speed_error

# #         if speed > 2.0:
# #             reward += 0.01

# #         if speed < 1.0:
# #             reward -= 0.2

# #         if self.collision_sensor and len(self.collision_sensor.history) > 0:
# #             reward -= 30.0

# #         if self.lane_sensor and self.lane_sensor.count > 0:
# #             reward -= 1.0

# #         if dist_to_goal < 5.0 or progress >= 0.995:
# #             reward += 100.0

# #         return float(reward)

# #     def _check_done(self) -> Tuple[bool, bool, Dict]:
# #         assert self.ego is not None

# #         info = {}

# #         speed = self._kmh(self.ego.get_velocity())
# #         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
# #         collided = self.collision_sensor and len(self.collision_sensor.history) > 0

# #         terminated = False
# #         truncated = False

# #         if speed < 1.0:
# #             self.stuck_steps += 1
# #         else:
# #             self.stuck_steps = 0

# #         if collided:
# #             terminated = True
# #             info["event"] = "collision"

# #         elif abs(lateral_error) > 4.0:
# #             terminated = True
# #             info["event"] = "off_route"

# #         elif dist_to_goal < 5.0 or progress >= 0.995:
# #             terminated = True
# #             info["event"] = "route_completed"

# #         elif self.stuck_steps >= 150:
# #             truncated = True
# #             info["event"] = "stuck"

# #         elif self.step_count >= self.cfg.max_episode_steps:
# #             truncated = True
# #             info["event"] = "time_limit"

# #         info["progress"] = progress
# #         info["dist_to_goal"] = dist_to_goal
# #         info["route_index"] = self.route_index
# #         info["speed_kmh"] = speed
# #         info["stuck_steps"] = self.stuck_steps
# #         info["lateral_error"] = lateral_error
# #         info["heading_error"] = heading_error

# #         return terminated, truncated, info

# #     def _build_pedestrian_config(self):
# #         if not self.route_waypoints:
# #             raise RuntimeError(
# #                 "self.route_waypoints está vazio. "
# #                 "Chame _prepare_route() antes de _build_pedestrian_config()."
# #             )

# #         trigger_idx = max(5, len(self.route_waypoints) // 3)
# #         trigger_idx = min(trigger_idx, len(self.route_waypoints) - 1)

# #         trigger_tf = self.route_waypoints[trigger_idx]

# #         config = SimpleNamespace()
# #         config.trigger_points = [trigger_tf]
# #         config.route = [(wp, None) for wp in self.route_waypoints]

# #         # campos esperados pela infraestrutura do ScenarioRunner
# #         config.weather = carla.WeatherParameters(sun_altitude_angle=70.0)
# #         config.friction = None
# #         config.other_actors = []
# #         config.route_var_name = None
# #         config.subtype = None
# #         config.name = "PedestrianCrossingGym"
# #         config.town = self.route_config.town

# #         return config

# #     def _build_pedestrian_scenario(self):
# #         if self.world is None:
# #             raise RuntimeError("self.world é None ao criar o cenário.")
# #         if self.ego is None:
# #             raise RuntimeError("self.ego é None ao criar o cenário.")

# #         self.scenario_config = self._build_pedestrian_config()

# #         self.active_scenario = PedestrianCrossing(
# #             world=self.world,
# #             ego_vehicles=[self.ego],
# #             config=self.scenario_config,
# #             debug_mode=False,
# #             criteria_enable=False,
# #             timeout=60,
# #         )

# #     def reset(self, *, seed=None, options=None):
# #         super().reset(seed=seed)

# #         if self.client is None:
# #             self._connect()

# #         if not self.route_configs:
# #             self._load_route_configs()

# #         self.step_count = 0
# #         self.last_progress = 0.0
# #         self.route_index = 0
# #         self.stuck_steps = 0
# #         self.prev_dist_to_goal = None

# #         old_town = self.route_config.town if self.route_config is not None else None

# #         self._choose_route()
# #         new_town = self.route_config.town

# #         self._cleanup_actors()

# #         if self.world is None or old_town != new_town:
# #             self._load_world_for_route()
# #         else:
# #             for _ in range(5):
# #                 self.world.tick()

# #         self._prepare_route()
# #         self._spawn_ego()
# #         self._warmup_ticks()
# #         self._update_spectator()

# #         self._build_pedestrian_scenario()

# #         if self.collision_sensor:
# #             self.collision_sensor.clear()
# #         if self.lane_sensor:
# #             self.lane_sensor.clear()

# #         self.last_location = self.ego.get_location()

# #         _, _, progress, dist_to_goal = self._compute_route_errors()
# #         self.last_progress = progress
# #         self.prev_dist_to_goal = dist_to_goal

# #         obs = self._get_obs()
# #         info = {
# #             "town": self.route_config.town,
# #             "route_name": getattr(self.route_config, "name", "unknown"),
# #             "route_length": len(self.route_waypoints),
# #         }
# #         return obs, info

# #     def step(self, action):
# #         assert self.ego is not None
# #         assert self.world is not None

# #         action = np.asarray(action, dtype=np.float32)
# #         steer = float(np.clip(action[0], -1.0, 1.0))
# #         accel = float(np.clip(action[1], -1.0, 1.0))

# #         if accel > 0.0:
# #             throttle = accel
# #             brake = 0.0
# #         else:
# #             throttle = 0.0
# #             brake = -accel

# #         control = carla.VehicleControl(
# #             steer=steer,
# #             throttle=throttle,
# #             brake=brake,
# #             hand_brake=False,
# #             reverse=False,
# #             manual_gear_shift=False,
# #         )

# #         self.ego.apply_control(control)

# #         self.world.tick()
# #         self.step_count += 1

# #         self._update_spectator()

# #         if self.active_scenario is not None:
# #             try:
# #                 self.active_scenario.scenario.scenario_tree.tick_once()
# #             except Exception:
# #                 pass

# #         obs = self._get_obs()
# #         reward = self._compute_reward()
# #         terminated, truncated, info = self._check_done()

# #         if self.cfg.show_bev and self.bev_camera is not None:
# #             bev = self.bev_camera.get_latest()
# #             if bev is not None:
# #                 if not self._bev_window_created:
# #                     cv2.namedWindow("CARLA Bird-Eye View", cv2.WINDOW_NORMAL)
# #                     self._bev_window_created = True

# #                 cv2.imshow("CARLA Bird-Eye View", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
# #                 cv2.waitKey(1)

# #         if self.cfg.render_rgb_camera and self.camera_sensor:
# #             rgb = self.camera_sensor.get_latest()
# #             if rgb is not None:
# #                 # só visualiza se quiser
# #                 pass

# #         return obs, reward, terminated, truncated, info

# #     def render(self):
# #         pass

# #     def close(self):
# #         self._cleanup_actors()

# #         if self.world is not None and self.cfg.sync:
# #             settings = self.world.get_settings()
# #             settings.synchronous_mode = False
# #             settings.fixed_delta_seconds = None
# #             self.world.apply_settings(settings)

# #         if self.traffic_manager is not None and self.cfg.sync:
# #             try:
# #                 self.traffic_manager.set_synchronous_mode(False)
# #             except RuntimeError:
# #                 pass

# #         cv2.destroyAllWindows()
# #################################################################################################

# import math
# import queue
# import random
# import weakref
# from types import SimpleNamespace
# from typing import Dict, List, Optional, Tuple

# import carla
# import cv2
# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces

# from srunner.tools.route_parser import RouteParser
# from srunner.tools.route_manipulation import interpolate_trajectory
# from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
# from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


# class EnvConfig:
#     host = "127.0.0.1"
#     port = 2000
#     traffic_manager_port = 8000
#     timeout = 10.0

#     route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
#     scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"

#     # Se route_id != None, usa uma rota específica do XML
#     route_id = None
#     # Exemplo:
#     # route_id = "20"

#     # Se route_town != None e route_id == None, escolhe aleatoriamente uma rota dessa town
#     route_town = None
#     # Exemplo:
#     # route_town = "Town05"

#     sync = True
#     fixed_delta_seconds = 0.05

#     max_episode_steps = 500
#     target_speed_kmh = 30.0

#     ego_filter = "vehicle.lincoln.mkz_2017"
#     seed = 42

#     render_rgb_camera = False

#     spectator_follow = True
#     spectator_height_m = 40.0
#     spectator_rotate_with_ego = False

#     # Recomendo deixar False no treino e True só quando quiser visualizar
#     show_bev = False
#     bev_width = 800
#     bev_height = 800
#     bev_fov = 90
#     bev_height_m = 35.0


# class CollisionSensor:
#     def __init__(self, parent_actor: carla.Actor):
#         self.sensor = None
#         self.history = []
#         self._parent = parent_actor

#         world = self._parent.get_world()
#         bp = world.get_blueprint_library().find("sensor.other.collision")
#         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

#         weak_self = weakref.ref(self)
#         self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

#     @staticmethod
#     def _on_collision(weak_self, event):
#         self = weak_self()
#         if not self:
#             return

#         impulse = event.normal_impulse
#         intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
#         self.history.append((event.frame, intensity))

#     def clear(self):
#         self.history.clear()

#     def destroy(self):
#         if self.sensor is not None:
#             self.sensor.stop()
#             self.sensor.destroy()
#             self.sensor = None


# class LaneInvasionSensor:
#     def __init__(self, parent_actor: carla.Actor):
#         self.sensor = None
#         self.count = 0
#         self._parent = parent_actor

#         world = self._parent.get_world()
#         bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
#         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

#         weak_self = weakref.ref(self)
#         self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

#     @staticmethod
#     def _on_invasion(weak_self, event):
#         self = weak_self()
#         if not self:
#             return
#         self.count += 1

#     def clear(self):
#         self.count = 0

#     def destroy(self):
#         if self.sensor is not None:
#             self.sensor.stop()
#             self.sensor.destroy()
#             self.sensor = None


# class CameraSensor:
#     def __init__(self, parent_actor: carla.Actor, width=640, height=360):
#         self.sensor = None
#         self.queue = queue.Queue()
#         self.width = width
#         self.height = height

#         world = parent_actor.get_world()
#         bp = world.get_blueprint_library().find("sensor.camera.rgb")
#         bp.set_attribute("image_size_x", str(width))
#         bp.set_attribute("image_size_y", str(height))
#         bp.set_attribute("fov", "90")
#         bp.set_attribute("sensor_tick", "0.0")

#         transform = carla.Transform(carla.Location(x=1.5, z=2.4))
#         self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
#         self.sensor.listen(self.queue.put)

#     def get_latest(self) -> Optional[np.ndarray]:
#         img = None
#         while not self.queue.empty():
#             img = self.queue.get()

#         if img is None:
#             return None

#         array = np.frombuffer(img.raw_data, dtype=np.uint8)
#         array = array.reshape((self.height, self.width, 4))
#         array = array[:, :, :3]
#         return array

#     def destroy(self):
#         if self.sensor is not None:
#             self.sensor.stop()
#             self.sensor.destroy()
#             self.sensor = None


# class BirdEyeCamera:
#     """
#     Câmera top-down presa ao ego.
#     Como está attach_to=parent_actor com AttachmentType.Rigid,
#     ela acompanha a posição e a rotação do carro.
#     """

#     def __init__(self, parent_actor: carla.Actor, width=800, height=800, fov=90, z=35.0):
#         self.sensor = None
#         self.queue = queue.Queue()
#         self.width = width
#         self.height = height

#         world = parent_actor.get_world()
#         bp = world.get_blueprint_library().find("sensor.camera.rgb")
#         bp.set_attribute("image_size_x", str(width))
#         bp.set_attribute("image_size_y", str(height))
#         bp.set_attribute("fov", str(fov))
#         bp.set_attribute("sensor_tick", "0.0")

#         transform = carla.Transform(
#             carla.Location(x=0.0, y=0.0, z=z),
#             carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
#         )

#         self.sensor = world.spawn_actor(
#             bp,
#             transform,
#             attach_to=parent_actor,
#             attachment_type=carla.AttachmentType.Rigid,
#         )
#         self.sensor.listen(self.queue.put)

#     def get_latest(self) -> Optional[np.ndarray]:
#         img = None
#         while not self.queue.empty():
#             img = self.queue.get()

#         if img is None:
#             return None

#         array = np.frombuffer(img.raw_data, dtype=np.uint8)
#         array = array.reshape((self.height, self.width, 4))
#         array = array[:, :, :3]
#         return array

#     def destroy(self):
#         if self.sensor is not None:
#             self.sensor.stop()
#             self.sensor.destroy()
#             self.sensor = None


# class CarlaRouteEnv(gym.Env):
#     def __init__(self, cfg: EnvConfig):
#         super().__init__()
#         self.cfg = cfg
#         self._rng = random.Random(cfg.seed)

#         # action[0] = steer em [-1, 1]
#         # action[1] = accel em [-1, 1]
#         #   accel > 0 -> throttle
#         #   accel < 0 -> brake
#         self.action_space = spaces.Box(
#             low=np.array([-1.0, -1.0], dtype=np.float32),
#             high=np.array([1.0, 1.0], dtype=np.float32),
#             dtype=np.float32,
#         )

#         # [speed, target_speed, lateral_error, heading_error,
#         #  progress, dist_to_goal, throttle, brake, steer,
#         #  lane_count, collided, accel_x, accel_y, ang_vel_z]
#         self.observation_space = spaces.Box(
#             low=-np.inf,
#             high=np.inf,
#             shape=(14,),
#             dtype=np.float32,
#         )

#         self.client: Optional[carla.Client] = None
#         self.world: Optional[carla.World] = None
#         self.map: Optional[carla.Map] = None
#         self.traffic_manager = None

#         self.route_configs = []
#         self.route_config = None
#         self.route_dense: List[Tuple[carla.Transform, object]] = []
#         self.route_waypoints: List[carla.Transform] = []
#         self.route_index = 0

#         self.ego: Optional[carla.Vehicle] = None
#         self.collision_sensor: Optional[CollisionSensor] = None
#         self.lane_sensor: Optional[LaneInvasionSensor] = None
#         self.camera_sensor: Optional[CameraSensor] = None
#         self.bev_camera: Optional[BirdEyeCamera] = None

#         self.actor_handles: List[carla.Actor] = []

#         self.step_count = 0
#         self.last_progress = 0.0
#         self.last_location: Optional[carla.Location] = None
#         self.stuck_steps = 0
#         self.prev_dist_to_goal = None

#         self.active_scenario = None
#         self.scenario_config = None

#         self._bev_window_created = False

#     def _connect(self):
#         self.client = carla.Client(self.cfg.host, self.cfg.port)
#         self.client.set_timeout(self.cfg.timeout)
#         self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)

#         CarlaDataProvider.set_client(self.client)
#         if hasattr(CarlaDataProvider, "set_traffic_manager_port"):
#             CarlaDataProvider.set_traffic_manager_port(self.cfg.traffic_manager_port)

#     def _load_route_configs(self):
#         self.route_configs = RouteParser.parse_routes_file(
#             self.cfg.route_file,
#             self.cfg.scenario_file,
#             single_route=self.cfg.route_id,
#         )

#         if not self.route_configs:
#             raise RuntimeError("No route found in the XML file.")

#     def _choose_route(self):
#         if self.cfg.route_id is not None:
#             self.route_config = self.route_configs[0]
#             return

#         route_town = getattr(self.cfg, "route_town", None)

#         if route_town is not None:
#             valid_routes = [rc for rc in self.route_configs if rc.town == route_town]
#             if not valid_routes:
#                 raise RuntimeError(f"No route found for town {route_town}")
#             self.route_config = self._rng.choice(valid_routes)
#         else:
#             self.route_config = self._rng.choice(self.route_configs)

#     def _load_world_for_route(self):
#         assert self.client is not None
#         assert self.route_config is not None

#         town = self.route_config.town
#         self.world = self.client.load_world(town)
#         self.map = self.world.get_map()
#         CarlaDataProvider.set_world(self.world)

#         if self.cfg.sync:
#             settings = self.world.get_settings()
#             settings.synchronous_mode = True
#             settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
#             self.world.apply_settings(settings)

#             self.client.reload_world(False)
#             self.world = self.client.get_world()
#             self.map = self.world.get_map()
#             CarlaDataProvider.set_world(self.world)

#             if self.traffic_manager is not None:
#                 self.traffic_manager.set_synchronous_mode(True)

#     def _cleanup_actors(self):
#         if self.active_scenario is not None:
#             try:
#                 self.active_scenario.remove_all_actors()
#             except Exception:
#                 pass
#             self.active_scenario = None
#             self.scenario_config = None

#         if self.camera_sensor:
#             self.camera_sensor.destroy()
#             self.camera_sensor = None

#         if self.collision_sensor:
#             self.collision_sensor.destroy()
#             self.collision_sensor = None

#         if self.lane_sensor:
#             self.lane_sensor.destroy()
#             self.lane_sensor = None

#         if self.bev_camera:
#             self.bev_camera.destroy()
#             self.bev_camera = None

#         for actor in self.actor_handles:
#             try:
#                 actor.destroy()
#             except RuntimeError:
#                 pass
#         self.actor_handles.clear()

#         self.ego = None

#         if hasattr(CarlaDataProvider, "cleanup"):
#             try:
#                 CarlaDataProvider.cleanup()
#             except Exception:
#                 pass

#     def _prepare_route(self):
#         assert self.world is not None
#         assert self.route_config is not None

#         _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
#         self.route_dense = dense_route
#         self.route_waypoints = [wp[0] for wp in dense_route]

#         if len(self.route_waypoints) < 2:
#             raise RuntimeError("The interpolated route was too short.")

#     def _spawn_ego(self):
#         assert self.world is not None
#         assert self.map is not None
#         assert self.route_config is not None

#         bp_lib = self.world.get_blueprint_library()
#         ego_bp = bp_lib.find(self.cfg.ego_filter)
#         ego_bp.set_attribute("role_name", "hero")

#         start_loc = self.route_config.trajectory[0]
#         start_wp = self.map.get_waypoint(start_loc)
#         spawn_transform = start_wp.transform
#         spawn_transform.location.z += 0.5

#         self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

#         if self.ego is None:
#             spawn_points = self.map.get_spawn_points()
#             self._rng.shuffle(spawn_points)
#             for sp in spawn_points:
#                 self.ego = self.world.try_spawn_actor(ego_bp, sp)
#                 if self.ego is not None:
#                     break

#         if self.ego is None:
#             raise RuntimeError("Failed to spawn the ego vehicle.")

#         self.actor_handles.append(self.ego)

#         self.collision_sensor = CollisionSensor(self.ego)
#         self.lane_sensor = LaneInvasionSensor(self.ego)

#         if self.cfg.render_rgb_camera:
#             self.camera_sensor = CameraSensor(self.ego)

#         if self.cfg.show_bev:
#             self.bev_camera = BirdEyeCamera(
#                 self.ego,
#                 width=self.cfg.bev_width,
#                 height=self.cfg.bev_height,
#                 fov=self.cfg.bev_fov,
#                 z=self.cfg.bev_height_m,
#             )

#     def _warmup_ticks(self, n=10):
#         assert self.world is not None
#         for _ in range(n):
#             self.world.tick()

#     def _kmh(self, vel: carla.Vector3D) -> float:
#         return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

#     def _distance(self, a: carla.Location, b: carla.Location) -> float:
#         return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

#     def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
#         start = max(0, self.route_index - 5)
#         end = min(len(self.route_waypoints), self.route_index + window)

#         best_idx = self.route_index
#         best_dist = float("inf")

#         for i in range(start, end):
#             d = self._distance(location, self.route_waypoints[i].location)
#             if d < best_dist:
#                 best_dist = d
#                 best_idx = i

#         return best_idx

#     def _compute_route_errors(self) -> Tuple[float, float, float, float]:
#         assert self.ego is not None
#         assert len(self.route_waypoints) > 0

#         ego_tf = self.ego.get_transform()
#         ego_loc = ego_tf.location
#         ego_yaw = math.radians(ego_tf.rotation.yaw)

#         self.route_index = self._find_nearest_route_index(ego_loc)
#         wp_tf = self.route_waypoints[self.route_index]
#         wp_loc = wp_tf.location
#         wp_yaw = math.radians(wp_tf.rotation.yaw)

#         dx = ego_loc.x - wp_loc.x
#         dy = ego_loc.y - wp_loc.y

#         lateral_error = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

#         heading_error = ego_yaw - wp_yaw
#         heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

#         progress = self.route_index / max(1, (len(self.route_waypoints) - 1))
#         dist_to_goal = self._distance(ego_loc, self.route_waypoints[-1].location)

#         return lateral_error, heading_error, progress, dist_to_goal

#     def _update_spectator(self):
#         if not self.cfg.spectator_follow or self.ego is None or self.world is None:
#             return

#         spectator = self.world.get_spectator()
#         ego_tf = self.ego.get_transform()
#         ego_loc = ego_tf.location

#         yaw = ego_tf.rotation.yaw if self.cfg.spectator_rotate_with_ego else 0.0

#         spec_tf = carla.Transform(
#             carla.Location(x=ego_loc.x, y=ego_loc.y, z=self.cfg.spectator_height_m),
#             carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
#         )
#         spectator.set_transform(spec_tf)

#     def _get_obs(self) -> np.ndarray:
#         assert self.ego is not None

#         vel = self.ego.get_velocity()
#         acc = self.ego.get_acceleration()
#         ang = self.ego.get_angular_velocity()
#         ctrl = self.ego.get_control()

#         speed = self._kmh(vel)
#         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

#         collided = 1.0 if (self.collision_sensor and len(self.collision_sensor.history) > 0) else 0.0
#         lane_count = float(self.lane_sensor.count if self.lane_sensor else 0.0)

#         obs = np.array(
#             [
#                 speed / 100.0,
#                 self.cfg.target_speed_kmh / 100.0,
#                 lateral_error / 10.0,
#                 heading_error / math.pi,
#                 progress,
#                 dist_to_goal / 1000.0,
#                 float(ctrl.throttle),
#                 float(ctrl.brake),
#                 float(ctrl.steer),
#                 min(lane_count, 10.0) / 10.0,
#                 collided,
#                 float(acc.x) / 10.0,
#                 float(acc.y) / 10.0,
#                 float(ang.z) / 10.0,
#             ],
#             dtype=np.float32,
#         )
#         return obs

#     def _compute_reward(self) -> float:
#         assert self.ego is not None

#         speed = self._kmh(self.ego.get_velocity())
#         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

#         progress_delta = progress - self.last_progress
#         self.last_progress = progress

#         if self.prev_dist_to_goal is None:
#             dist_delta = 0.0
#         else:
#             dist_delta = self.prev_dist_to_goal - dist_to_goal
#         self.prev_dist_to_goal = dist_to_goal

#         reward = 0.0

#         reward += 30.0 * progress_delta
#         reward += 0.05 * dist_delta

#         reward -= 0.05 * abs(lateral_error)
#         reward -= 0.05 * abs(heading_error)

#         speed_error = abs(speed - self.cfg.target_speed_kmh)
#         reward -= 0.02 * speed_error

#         if speed > 2.0:
#             reward += 0.01

#         if speed < 1.0:
#             reward -= 0.2

#         if self.collision_sensor and len(self.collision_sensor.history) > 0:
#             reward -= 30.0

#         if self.lane_sensor and self.lane_sensor.count > 0:
#             reward -= 1.0

#         if dist_to_goal < 5.0 or progress >= 0.995:
#             reward += 100.0

#         return float(reward)

#     def _check_done(self) -> Tuple[bool, bool, Dict]:
#         assert self.ego is not None

#         info = {}

#         speed = self._kmh(self.ego.get_velocity())
#         lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
#         collided = self.collision_sensor and len(self.collision_sensor.history) > 0

#         terminated = False
#         truncated = False

#         if speed < 1.0:
#             self.stuck_steps += 1
#         else:
#             self.stuck_steps = 0

#         if collided:
#             terminated = True
#             info["event"] = "collision"

#         elif abs(lateral_error) > 4.0:
#             terminated = True
#             info["event"] = "off_route"

#         elif dist_to_goal < 5.0 or progress >= 0.995:
#             terminated = True
#             info["event"] = "route_completed"

#         elif self.stuck_steps >= 150:
#             truncated = True
#             info["event"] = "stuck"

#         elif self.step_count >= self.cfg.max_episode_steps:
#             truncated = True
#             info["event"] = "time_limit"

#         info["progress"] = progress
#         info["dist_to_goal"] = dist_to_goal
#         info["route_index"] = self.route_index
#         info["speed_kmh"] = speed
#         info["stuck_steps"] = self.stuck_steps
#         info["lateral_error"] = lateral_error
#         info["heading_error"] = heading_error

#         return terminated, truncated, info

#     def _build_pedestrian_config(self):
#         if not self.route_waypoints:
#             raise RuntimeError(
#                 "self.route_waypoints está vazio. "
#                 "Chame _prepare_route() antes de _build_pedestrian_config()."
#             )

#         trigger_idx = max(5, len(self.route_waypoints) // 3)
#         trigger_idx = min(trigger_idx, len(self.route_waypoints) - 1)
#         trigger_tf = self.route_waypoints[trigger_idx]

#         config = SimpleNamespace()
#         config.trigger_points = [trigger_tf]
#         config.route = [(wp, None) for wp in self.route_waypoints]

#         config.weather = carla.WeatherParameters(sun_altitude_angle=70.0)
#         config.friction = None
#         config.other_actors = []
#         config.route_var_name = None
#         config.subtype = None
#         config.name = "PedestrianCrossingGym"
#         config.town = self.route_config.town

#         return config

#     def _build_pedestrian_scenario(self):
#         if self.world is None:
#             raise RuntimeError("self.world é None ao criar o cenário.")
#         if self.ego is None:
#             raise RuntimeError("self.ego é None ao criar o cenário.")

#         CarlaDataProvider.set_world(self.world)

#         self.scenario_config = self._build_pedestrian_config()

#         try:
#             self.active_scenario = PedestrianCrossing(
#                 world=self.world,
#                 ego_vehicles=[self.ego],
#                 config=self.scenario_config,
#                 debug_mode=False,
#                 criteria_enable=False,
#                 timeout=60,
#             )
#         except Exception:
#             self.active_scenario = None
#             raise

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)

#         if self.client is None:
#             self._connect()

#         if not self.route_configs:
#             self._load_route_configs()

#         self.step_count = 0
#         self.last_progress = 0.0
#         self.route_index = 0
#         self.stuck_steps = 0
#         self.prev_dist_to_goal = None

#         old_town = self.route_config.town if self.route_config is not None else None

#         self._choose_route()
#         new_town = self.route_config.town

#         self._cleanup_actors()

#         if self.world is None or old_town != new_town:
#             self._load_world_for_route()
#         else:
#             CarlaDataProvider.set_world(self.world)
#             for _ in range(5):
#                 self.world.tick()

#         self._prepare_route()
#         self._spawn_ego()
#         self._warmup_ticks()
#         self._update_spectator()

#         self._build_pedestrian_scenario()

#         if self.collision_sensor:
#             self.collision_sensor.clear()
#         if self.lane_sensor:
#             self.lane_sensor.clear()

#         self.last_location = self.ego.get_location()

#         _, _, progress, dist_to_goal = self._compute_route_errors()
#         self.last_progress = progress
#         self.prev_dist_to_goal = dist_to_goal

#         obs = self._get_obs()
#         info = {
#             "town": self.route_config.town,
#             "route_name": getattr(self.route_config, "name", "unknown"),
#             "route_length": len(self.route_waypoints),
#         }
#         return obs, info

#     def step(self, action):
#         assert self.ego is not None
#         assert self.world is not None

#         action = np.asarray(action, dtype=np.float32)
#         steer = float(np.clip(action[0], -1.0, 1.0))
#         accel = float(np.clip(action[1], -1.0, 1.0))

#         if accel > 0.0:
#             throttle = accel
#             brake = 0.0
#         else:
#             throttle = 0.0
#             brake = -accel

#         control = carla.VehicleControl(
#             steer=steer,
#             throttle=throttle,
#             brake=brake,
#             hand_brake=False,
#             reverse=False,
#             manual_gear_shift=False,
#         )
#         self.ego.apply_control(control)

#         self.world.tick()
#         self.step_count += 1

#         self._update_spectator()

#         if self.active_scenario is not None:
#             try:
#                 self.active_scenario.scenario.scenario_tree.tick_once()
#             except Exception:
#                 pass

#         obs = self._get_obs()
#         reward = self._compute_reward()
#         terminated, truncated, info = self._check_done()

#         if self.cfg.show_bev and self.bev_camera is not None:
#             bev = self.bev_camera.get_latest()
#             if bev is not None:
#                 if not self._bev_window_created:
#                     cv2.namedWindow("CARLA Bird-Eye View", cv2.WINDOW_NORMAL)
#                     self._bev_window_created = True

#                 cv2.imshow("CARLA Bird-Eye View", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
#                 cv2.waitKey(1)

#         return obs, reward, terminated, truncated, info

#     def render(self):
#         pass

#     def close(self):
#         self._cleanup_actors()

#         if self.world is not None and self.cfg.sync:
#             settings = self.world.get_settings()
#             settings.synchronous_mode = False
#             settings.fixed_delta_seconds = None
#             self.world.apply_settings(settings)

#         if self.traffic_manager is not None and self.cfg.sync:
#             try:
#                 self.traffic_manager.set_synchronous_mode(False)
#             except RuntimeError:
#                 pass

#         cv2.destroyAllWindows()

import math
import queue
import random
import weakref
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from srunner.tools.route_parser import RouteParser
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class EnvConfig:
    host = "127.0.0.1"
    port = 2000
    traffic_manager_port = 8000
    timeout = 10.0

    # route_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/routes_devtest.xml"
    # scenario_file = "/home/ana/Documents/Architecture_Transformers_SR/scenario_runner/srunner/data/all_towns_traffic_scenarios1_3_4.json"
    scenario_file = None

    route_id = None
    route_town = None

    sync = True
    fixed_delta_seconds = 0.05

    max_episode_steps = 500
    target_speed_kmh = 30.0

    ego_filter = "vehicle.lincoln.mkz_2017"
    seed = 42

    # Para debug visual
    render_rgb_camera = True

    spectator_follow = True
    spectator_height_m = 12.0
    spectator_rotate_with_ego = True

    show_bev = True
    bev_width = 800
    bev_height = 800
    bev_fov = 90
    bev_height_m = 35.0


class CollisionSensor:
    def __init__(self, parent_actor: carla.Actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))

    def clear(self):
        self.history.clear()

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class LaneInvasionSensor:
    def __init__(self, parent_actor: carla.Actor):
        self.sensor = None
        self.count = 0
        self._parent = parent_actor

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.count += 1

    def clear(self):
        self.count = 0

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class CameraSensor:
    def __init__(self, parent_actor: carla.Actor, width=640, height=360):
        self.sensor = None
        self.queue = queue.Queue()
        self.width = width
        self.height = height

        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")
        bp.set_attribute("sensor_tick", "0.0")

        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
        self.sensor.listen(self.queue.put)

    def get_latest(self) -> Optional[np.ndarray]:
        img = None
        while not self.queue.empty():
            img = self.queue.get()

        if img is None:
            return None

        array = np.frombuffer(img.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))
        array = array[:, :, :3]
        return array

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class BirdEyeCamera:
    def __init__(self, parent_actor: carla.Actor, width=800, height=800, fov=90, z=35.0):
        self.sensor = None
        self.queue = queue.Queue()
        self.width = width
        self.height = height

        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.0")

        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=z),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )

        self.sensor = world.spawn_actor(
            bp,
            transform,
            attach_to=parent_actor,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.sensor.listen(self.queue.put)

    def get_latest(self) -> Optional[np.ndarray]:
        img = None
        while not self.queue.empty():
            img = self.queue.get()

        if img is None:
            return None

        array = np.frombuffer(img.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))
        array = array[:, :, :3]
        return array

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None


class CarlaRouteEnv(gym.Env):
    def __init__(self, cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.traffic_manager = None

        self.route_configs = []
        self.route_config = None
        self.route_dense: List[Tuple[carla.Transform, object]] = []
        self.route_waypoints: List[carla.Transform] = []
        self.route_index = 0

        self.ego: Optional[carla.Vehicle] = None
        self.collision_sensor: Optional[CollisionSensor] = None
        self.lane_sensor: Optional[LaneInvasionSensor] = None
        self.camera_sensor: Optional[CameraSensor] = None
        self.bev_camera: Optional[BirdEyeCamera] = None

        self.actor_handles: List[carla.Actor] = []

        self.step_count = 0
        self.last_progress = 0.0
        self.last_location: Optional[carla.Location] = None
        self.stuck_steps = 0
        self.prev_dist_to_goal = None

        self.active_scenario = None
        self.scenario_config = None

        self._bev_window_created = False
        self._rgb_window_created = False

    def _connect(self):
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout)
        self.traffic_manager = self.client.get_trafficmanager(self.cfg.traffic_manager_port)

        CarlaDataProvider.set_client(self.client)
        if hasattr(CarlaDataProvider, "set_traffic_manager_port"):
            CarlaDataProvider.set_traffic_manager_port(self.cfg.traffic_manager_port)

    def _load_route_configs(self):
        self.route_configs = RouteParser.parse_routes_file(
            self.cfg.route_file,
            self.cfg.scenario_file,
            single_route=self.cfg.route_id,
        )

        if not self.route_configs:
            raise RuntimeError("No route found in the XML file.")

    def _choose_route(self):
        if self.cfg.route_id is not None:
            self.route_config = self.route_configs[0]
            return

        route_town = getattr(self.cfg, "route_town", None)

        if route_town is not None:
            valid_routes = [rc for rc in self.route_configs if rc.town == route_town]
            if not valid_routes:
                raise RuntimeError(f"No route found for town {route_town}")
            self.route_config = self._rng.choice(valid_routes)
        else:
            self.route_config = self._rng.choice(self.route_configs)

    def _load_world_for_route(self):
        assert self.client is not None
        assert self.route_config is not None

        town = self.route_config.town
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        CarlaDataProvider.set_world(self.world)

        if self.cfg.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self.world.apply_settings(settings)

            self.client.reload_world(False)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            CarlaDataProvider.set_world(self.world)

            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(True)

    def _cleanup_actors(self):
        if self.active_scenario is not None:
            try:
                if hasattr(self.active_scenario, "terminate"):
                    self.active_scenario.terminate()
            except Exception:
                pass

            try:
                self.active_scenario.remove_all_actors()
            except Exception:
                pass

            self.active_scenario = None
            self.scenario_config = None

        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None

        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.lane_sensor:
            self.lane_sensor.destroy()
            self.lane_sensor = None

        if self.bev_camera:
            self.bev_camera.destroy()
            self.bev_camera = None

        for actor in self.actor_handles:
            try:
                actor.destroy()
            except RuntimeError:
                pass
            except Exception:
                pass
        self.actor_handles.clear()

        self.ego = None

        if hasattr(CarlaDataProvider, "cleanup"):
            try:
                CarlaDataProvider.cleanup()
            except Exception:
                pass

    def _prepare_route(self):
        assert self.world is not None
        assert self.route_config is not None

        _, dense_route = interpolate_trajectory(self.world, self.route_config.trajectory)
        self.route_dense = dense_route
        self.route_waypoints = [wp[0] for wp in dense_route]

        if len(self.route_waypoints) < 2:
            raise RuntimeError("The interpolated route was too short.")

    def _spawn_ego(self):
        assert self.world is not None
        assert self.map is not None
        assert self.route_config is not None

        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find(self.cfg.ego_filter)
        ego_bp.set_attribute("role_name", "hero")

        start_loc = self.route_config.trajectory[0]
        start_wp = self.map.get_waypoint(start_loc)
        spawn_transform = start_wp.transform
        spawn_transform.location.z += 0.5

        self.ego = self.world.try_spawn_actor(ego_bp, spawn_transform)

        if self.ego is None:
            spawn_points = self.map.get_spawn_points()
            self._rng.shuffle(spawn_points)
            for sp in spawn_points:
                self.ego = self.world.try_spawn_actor(ego_bp, sp)
                if self.ego is not None:
                    break

        if self.ego is None:
            raise RuntimeError("Failed to spawn the ego vehicle.")

        self.actor_handles.append(self.ego)

        self.collision_sensor = CollisionSensor(self.ego)
        self.lane_sensor = LaneInvasionSensor(self.ego)

        if self.cfg.render_rgb_camera:
            self.camera_sensor = CameraSensor(self.ego)

        if self.cfg.show_bev:
            self.bev_camera = BirdEyeCamera(
                self.ego,
                width=self.cfg.bev_width,
                height=self.cfg.bev_height,
                fov=self.cfg.bev_fov,
                z=self.cfg.bev_height_m,
            )

    def _warmup_ticks(self, n=10):
        assert self.world is not None
        for _ in range(n):
            self.world.tick()
            try:
                CarlaDataProvider.on_carla_tick()
            except Exception:
                pass

    def _kmh(self, vel: carla.Vector3D) -> float:
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def _distance(self, a: carla.Location, b: carla.Location) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _find_nearest_route_index(self, location: carla.Location, window: int = 40) -> int:
        start = max(0, self.route_index - 5)
        end = min(len(self.route_waypoints), self.route_index + window)

        best_idx = self.route_index
        best_dist = float("inf")

        for i in range(start, end):
            d = self._distance(location, self.route_waypoints[i].location)
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def _compute_route_errors(self) -> Tuple[float, float, float, float]:
        assert self.ego is not None
        assert len(self.route_waypoints) > 0

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        self.route_index = self._find_nearest_route_index(ego_loc)
        wp_tf = self.route_waypoints[self.route_index]
        wp_loc = wp_tf.location
        wp_yaw = math.radians(wp_tf.rotation.yaw)

        dx = ego_loc.x - wp_loc.x
        dy = ego_loc.y - wp_loc.y

        lateral_error = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy

        heading_error = ego_yaw - wp_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        progress = self.route_index / max(1, (len(self.route_waypoints) - 1))
        dist_to_goal = self._distance(ego_loc, self.route_waypoints[-1].location)

        return lateral_error, heading_error, progress, dist_to_goal

    def _update_spectator(self):
        if not self.cfg.spectator_follow or self.ego is None or self.world is None:
            return

        spectator = self.world.get_spectator()
        ego_tf = self.ego.get_transform()
        forward = ego_tf.get_forward_vector()

        cam_loc = carla.Location(
            x=ego_tf.location.x - 8.0 * forward.x,
            y=ego_tf.location.y - 8.0 * forward.y,
            z=4.0,
        )

        cam_rot = carla.Rotation(
            pitch=-15.0,
            yaw=ego_tf.rotation.yaw if self.cfg.spectator_rotate_with_ego else 0.0,
            roll=0.0,
        )

        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def _get_obs(self) -> np.ndarray:
        assert self.ego is not None

        vel = self.ego.get_velocity()
        acc = self.ego.get_acceleration()
        ang = self.ego.get_angular_velocity()
        ctrl = self.ego.get_control()

        speed = self._kmh(vel)
        lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

        collided = 1.0 if (self.collision_sensor and len(self.collision_sensor.history) > 0) else 0.0
        lane_count = float(self.lane_sensor.count if self.lane_sensor else 0.0)

        obs = np.array(
            [
                speed / 100.0,
                self.cfg.target_speed_kmh / 100.0,
                lateral_error / 10.0,
                heading_error / math.pi,
                progress,
                dist_to_goal / 1000.0,
                float(ctrl.throttle),
                float(ctrl.brake),
                float(ctrl.steer),
                min(lane_count, 10.0) / 10.0,
                collided,
                float(acc.x) / 10.0,
                float(acc.y) / 10.0,
                float(ang.z) / 10.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self) -> float:
        assert self.ego is not None

        speed = self._kmh(self.ego.get_velocity())
        lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()

        progress_delta = progress - self.last_progress
        self.last_progress = progress

        if self.prev_dist_to_goal is None:
            dist_delta = 0.0
        else:
            dist_delta = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        reward = 0.0
        reward += 30.0 * progress_delta
        reward += 0.05 * dist_delta
        reward -= 0.05 * abs(lateral_error)
        reward -= 0.05 * abs(heading_error)

        speed_error = abs(speed - self.cfg.target_speed_kmh)
        reward -= 0.02 * speed_error

        if speed > 2.0:
            reward += 0.01

        if speed < 1.0:
            reward -= 0.2

        if self.collision_sensor and len(self.collision_sensor.history) > 0:
            reward -= 30.0

        if self.lane_sensor and self.lane_sensor.count > 0:
            reward -= 1.0

        if dist_to_goal < 5.0 or progress >= 0.995:
            reward += 100.0

        return float(reward)

    def _check_done(self) -> Tuple[bool, bool, Dict]:
        assert self.ego is not None

        info = {}

        speed = self._kmh(self.ego.get_velocity())
        lateral_error, heading_error, progress, dist_to_goal = self._compute_route_errors()
        collided = self.collision_sensor and len(self.collision_sensor.history) > 0

        terminated = False
        truncated = False

        if speed < 1.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if collided:
            terminated = True
            info["event"] = "collision"
        elif abs(lateral_error) > 4.0:
            terminated = True
            info["event"] = "off_route"
        elif dist_to_goal < 5.0 or progress >= 0.995:
            terminated = True
            info["event"] = "route_completed"
        elif self.stuck_steps >= 150:
            truncated = True
            info["event"] = "stuck"
        elif self.step_count >= self.cfg.max_episode_steps:
            truncated = True
            info["event"] = "time_limit"

        info["progress"] = progress
        info["dist_to_goal"] = dist_to_goal
        info["route_index"] = self.route_index
        info["speed_kmh"] = speed
        info["stuck_steps"] = self.stuck_steps
        info["lateral_error"] = lateral_error
        info["heading_error"] = heading_error

        return terminated, truncated, info

    def _build_pedestrian_config(self):
        if not self.route_waypoints:
            raise RuntimeError("self.route_waypoints está vazio.")

        trigger_idx = max(5, len(self.route_waypoints) // 3)
        trigger_idx = min(trigger_idx, len(self.route_waypoints) - 1)
        trigger_tf = self.route_waypoints[trigger_idx]

        config = SimpleNamespace()
        config.trigger_points = [trigger_tf]
        config.route = [(wp, None) for wp in self.route_waypoints]

        config.weather = carla.WeatherParameters(sun_altitude_angle=70.0)
        config.friction = None
        config.other_actors = []
        config.route_var_name = None
        config.subtype = None
        config.name = "PedestrianCrossingGym"
        config.town = self.route_config.town

        return config

    def _build_pedestrian_scenario(self):
        if self.world is None:
            raise RuntimeError("self.world é None ao criar o cenário.")
        if self.ego is None:
            raise RuntimeError("self.ego é None ao criar o cenário.")

        CarlaDataProvider.set_world(self.world)

        self.scenario_config = self._build_pedestrian_config()

        try:
            self.active_scenario = PedestrianCrossing(world=self.world,
                                                    ego_vehicles=[self.ego],
                                                    config=self.scenario_config,
                                                    debug_mode=False,
                                                    criteria_enable=False,
                                                    timeout=60,
                                                    )
        except Exception:
            self.active_scenario = None
            raise

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.client is None:
            self._connect()

        if not self.route_configs:
            self._load_route_configs()

        self.step_count = 0
        self.last_progress = 0.0
        self.route_index = 0
        self.stuck_steps = 0
        self.prev_dist_to_goal = None

        old_town = self.route_config.town if self.route_config is not None else None

        self._choose_route()
        new_town = self.route_config.town

        self._cleanup_actors()

        if self.world is None or old_town != new_town:
            self._load_world_for_route()
        else:
            CarlaDataProvider.set_world(self.world)
            for _ in range(5):
                self.world.tick()
                try:
                    CarlaDataProvider.on_carla_tick()
                except Exception:
                    pass

        self._prepare_route()
        self._spawn_ego()
        self._warmup_ticks()
        self._update_spectator()

        self._build_pedestrian_scenario()

        try:
            CarlaDataProvider.on_carla_tick()
        except Exception:
            pass

        if self.active_scenario is not None:
            try:
                self.active_scenario.scenario.scenario_tree.tick_once()
            except Exception as e:
                print("[Scenario init tick error]", e)

        self.world.tick()
        try:
            CarlaDataProvider.on_carla_tick()
        except Exception:
            pass

        if self.collision_sensor:
            self.collision_sensor.clear()
        if self.lane_sensor:
            self.lane_sensor.clear()

        self.last_location = self.ego.get_location()

        _, _, progress, dist_to_goal = self._compute_route_errors()
        self.last_progress = progress
        self.prev_dist_to_goal = dist_to_goal

        obs = self._get_obs()
        info = {
            "town": self.route_config.town,
            "route_name": getattr(self.route_config, "name", "unknown"),
            "route_length": len(self.route_waypoints),
        }
        return obs, info

    def step(self, action):
        assert self.ego is not None
        assert self.world is not None

        action = np.asarray(action, dtype=np.float32)
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], -1.0, 1.0))

        if accel > 0.0:
            throttle = accel
            brake = 0.0
        else:
            throttle = 0.0
            brake = -accel

        control = carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
        self.ego.apply_control(control)

        self.world.tick()
        self.step_count += 1

        try:
            CarlaDataProvider.on_carla_tick()
        except Exception:
            pass

        self._update_spectator()

        if self.active_scenario is not None:
            try:
                self.active_scenario.scenario.scenario_tree.tick_once()
            except Exception as e:
                print("[Scenario tick error]", e)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated, info = self._check_done()

        if self.cfg.render_rgb_camera and self.camera_sensor is not None:
            rgb = self.camera_sensor.get_latest()
            if rgb is not None:
                if not self._rgb_window_created:
                    cv2.namedWindow("CARLA RGB Camera", cv2.WINDOW_NORMAL)
                    self._rgb_window_created = True

                cv2.imshow("CARLA RGB Camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        if self.cfg.show_bev and self.bev_camera is not None:
            bev = self.bev_camera.get_latest()
            if bev is not None:
                if not self._bev_window_created:
                    cv2.namedWindow("CARLA Bird-Eye View", cv2.WINDOW_NORMAL)
                    self._bev_window_created = True

                cv2.imshow("CARLA Bird-Eye View", cv2.cvtColor(bev, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._cleanup_actors()

        if self.world is not None and self.cfg.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        if self.traffic_manager is not None and self.cfg.sync:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass

        cv2.destroyAllWindows()