import carla
import numpy as np
import open3d as o3d
from PIL import Image

def depth_to_point_cloud(depth_map):
    fx = 2892.33
    fy = 2883.18
    cx = 823.205
    cy = 619.071

    h, w = depth_map.shape
    points = []
    for v in range(h):
        for u in range(w):
            Z = depth_map[v, u]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
    return np.array(points)

class UAV:
    def __init__(self, world, location, uav_id, yaw_angle=0):
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.uav_id = uav_id  # 添加UAV的唯一ID
        self.static_actor = None
        self.sensors = []
        self.ticks_per_capture = 2  # 每多少tick采集一次数据
        self.sensors_capture_intervals = self.ticks_per_capture * world.get_settings().fixed_delta_seconds  # 传感器的采集
        self.tick_counter = 0  # 初始化tick计数器
        self.move_enabled = False  # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 默认的位移向量
        self.noise_std = 0  # 随机扰动的标准差
        self.rgb_sensors_active = [False, False, False, False, True]
        self.depth_sensors_active = [False, False, False, False, False]
        self.spawn_uav()

    def spawn_uav(self):
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45
        fov = 90
        capture_intervals = self.sensors_capture_intervals  # 无人机的移动频率应该和传感器的采集频率一致

        directions = ["North", "East", "South", "West"]
        yaw_angles = [0, 90, 180, 270]

        static_blueprint = self.world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(self.location, carla.Rotation(yaw=self.yaw_angle))
        self.static_actor = self.world.spawn_actor(static_blueprint, spawn_point)

        # 创建垂直向下的传感器
        rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
        if self.rgb_sensors_active[4]:
            rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
        self.sensors.append(rgb_sensor)

        depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_blueprint.set_attribute('image_size_x', str(image_size_x))
        depth_blueprint.set_attribute('image_size_y', str(image_size_y))
        depth_blueprint.set_attribute('fov', str(fov))
        depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
        if self.depth_sensors_active[4]:
            depth_sensor.listen(lambda data: self.process_depth_image(data, "down", "depth"))
        self.sensors.append(depth_sensor)

        # 创建四个不同方向的传感器
        for direction, yaw in zip(directions, yaw_angles):
            adjusted_yaw = yaw - self.yaw_angle  # 根据无人机的朝向调整传感器的朝向

            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree))
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            if self.rgb_sensors_active[yaw // 90]:
                rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

            depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', str(image_size_x))
            depth_blueprint.set_attribute('image_size_y', str(image_size_y))
            depth_blueprint.set_attribute('fov', str(fov))
            depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree))
            depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
            if self.depth_sensors_active[yaw // 90]:
                depth_sensor.listen(lambda data, dir=direction: self.process_depth_image(data, dir, "depth"))
            self.sensors.append(depth_sensor)

    def process_image(self, image, direction, sensor_type):
        file_name = r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_uav%s_%s_%s_%06d.png' % (self.uav_id, direction, sensor_type, image.frame)
        image.save_to_disk(file_name)

    def process_depth_image(self, image, direction, sensor_type):
        file_name = r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_uav%s_%s_%s_%06d.png' % (self.uav_id, direction, sensor_type, image.frame)
        image.convert(carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(file_name)
        depth_map = Image.open(file_name).convert("L")
        depth_map = np.array(depth_map)
        points = depth_to_point_cloud(depth_map)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_file_name = r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_dot_out\dot_uav%s_%s_%s_%06d.pcd' % (self.uav_id, direction, sensor_type, image.frame)
        o3d.io.write_point_cloud(pcd_file_name, pcd)

    def move(self):
        """
        控制无人机移动，并添加随机扰动。
        只有在移动开关打开时才会移动，每隔指定的tick数进行一次移动。
        """
        noise = np.random.normal(0, self.noise_std, size=3)

        delta_location_with_noise = carla.Location(
            x=self.delta_location.x + noise[0],
            y=self.delta_location.y + noise[1],
            z=self.delta_location.z + noise[2]
        )

        new_location = self.static_actor.get_location() + delta_location_with_noise

        self.static_actor.set_location(new_location)

    def update(self):
        """
        每次tick调用该方法来检查是否需要移动无人机。
        """
        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_capture:
                self.tick_counter = 0
                self.move()

    def set_rgb_sensors_active(self, sensor_num, active=True):
        self.rgb_sensors_active[sensor_num] = active

    def set_depth_sensors_active(self, sensor_num, active=True):
        self.depth_sensors_active[sensor_num] = active

    def set_sensors_capture_intervals(self, intervals):
        self.sensors_capture_intervals = intervals

    def enable_movement(self, enabled=True):
        self.move_enabled = enabled

    def set_ticks_per_move(self, ticks):
        self.ticks_per_move = ticks

    def set_delta_location(self, delta_location):
        self.delta_location = delta_location

    def set_noise_std(self, noise_std):
        self.noise_std = noise_std

    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()
        if self.static_actor is not None:
            self.static_actor.destroy()
