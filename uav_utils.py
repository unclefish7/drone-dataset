"""
UAV Utils - CARLA 无人机数据采集工具

这个模块提供了在 CARLA 仿真环境中创建和控制无人机（UAV）的功能，用于采集多传感器数据和生成 BEV 语义分割图像。

主要功能：
1. 无人机生成和传感器配置
2. 多传感器数据采集（RGB相机、LiDAR、语义分割）
3. BEV 语义分割图像生成
4. 车辆信息检测和记录
5. 数据同步保存

工作流程：
1. 初始化 UAV 对象，配置传感器（RGB相机x5、LiDAR、语义分割传感器）
2. 传感器开始采集数据，数据存储在队列中
3. 每个 tick 调用 update() 方法：
   - check_and_save_all() 处理队列中的传感器数据
   - 当收集到所有传感器数据后，开始保存：
     - RGB图像保存为 PNG 格式
     - LiDAR数据保存为 PCD 格式，同时生成 BEV 可见性图像
     - 语义分割数据用于生成 5 种 BEV 图像
     - 车辆信息保存到 YAML 文件
4. 可选的移动功能：根据设定的参数移动 UAV 位置

BEV 图像类型（256x256，范围 -51.2m 到 51.2m）：
- bev_dynamic: 动态目标（车辆）的投影轮廓
- bev_static: 静态可行驶道路区域
- bev_lane: 车道线标记
- bev_visibility: 单个 UAV 的 LiDAR 实际可见范围（基于点云检测到的车辆）
- bev_visibility_corp: 多 UAV 协同可见范围（当前与单个 UAV 相同）

数据输出格式：
- {frame}_{camera_id}.png: RGB 图像
- {frame}.pcd: LiDAR 点云文件
- {frame}_bev_{type}.png: BEV 语义分割图像
- {frame}.yaml: 传感器参数和车辆信息

使用示例：
    uav = UAV(world, location, uav_id, root_dir)
    # 在主循环中调用
    uav.update(vehicles, current_tick)
    # 清理资源
    uav.destroy()
"""

import carla
import numpy as np
import open3d as o3d
import yaml
import os
import time
import math

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image
from queue import Queue
from queue import Empty

total_uav_num = 5

class RGBData:
    def __init__(self, sensor_id, data):
        self.sensor_id = sensor_id  # 传感器编号
        self.data = data              # RGB数据

class UAV:
    max_frame = 0
    def __init__(self, world, location, uav_id, root_dir, yaw_angle=0):
        """
        初始化 UAV 类。

        参数：
        - world：CARLA 的 world 对象。
        - location：无人机的初始位置（carla.Location 对象）。
        - uav_id：无人机的唯一 ID。
        - yaw_angle：无人机的初始偏航角。
        """
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.uav_id = uav_id  # 无人机的唯一 ID

        # 数据保存的根目录
        self.root_dir = root_dir
        self.self_dir = os.path.join(self.root_dir, str(self.uav_id))
        
        # 确保 UAV 对应的文件夹存在
        if not os.path.exists(self.self_dir):
            os.makedirs(self.self_dir)
            print(f"Created directory for UAV {self.uav_id}: {self.self_dir}")

        self.static_actor = None  # 静态演员，表示无人机的位置
        self.sensors = []         # 存储所有传感器的列表
        self.lidar_sensor = None  # 激光雷达传感器
        self.semantic_segmentation_sensor = None

        self.sensors_data_counter = 0  # 已接收到的传感器数据计数
        self.total_sensors = 0

        self.rgb_data_list = []  # 存储 RGB 数据的列表
        self.lidar_data = None

        self.sensor_queue = Queue()  # 传感器数据队列

        # 传感器采集间隔设置
        self.sensors_capture_intervals = 0.1  # 传感器采集间隔（秒）
        self.ticks_per_capture = self.sensors_capture_intervals / world.get_settings().fixed_delta_seconds
        self.tick_counter = 0  # tick 计数器

        # 计数器，所有传感器都存好数据以后才进行下一个tick
        self.if_saved = False

        # 移动设置
        self.move_enabled = False               # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 移动的位移向量
        self.noise_std = 0                      # 随机扰动的标准差

        # 传感器激活标志
        self.rgb_sensors_active = True  # 是否激活 RGB 相机传感器
        self.dot_sensors_active = True  # 是否激活激光雷达传感器
        self.segmentation_sensor_active = True  # 是否激活语义分割传感器

        if self.rgb_sensors_active:
            self.total_sensors += 5

        if self.dot_sensors_active:
            self.total_sensors += 1

        if self.segmentation_sensor_active:
            self.total_sensors += 1

        # 生成无人机并附加传感器
        self.spawn_uav()

    def spawn_uav(self):
        """
        生成无人机并附加传感器。
        """
        # 传感器参数设置
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45  # 相机俯仰角
        fov = 90            # 视场角
        capture_intervals = self.sensors_capture_intervals  # 采集间隔

        # 不同方向的设置
        directions = ["camera0", "camera1", "camera2", "camera3"]
        yaw_angles = [0, 90, 180, 270]
        sensor_offset = [
            [1, 0, -1],   # Front: 沿 x 正方向偏移
            [0, 1, -1],   # Right: 沿 y 正方向偏移
            [-1, 0, -1],  # Back: 沿 x 负方向偏移
            [0, -1, -1]   # Left: 沿 y 负方向偏移
        ]

        # 生成静态演员，表示无人机的位置
        static_blueprint = self.world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(self.location, carla.Rotation(yaw=self.yaw_angle))
        self.static_actor = self.world.spawn_actor(static_blueprint, spawn_point)

        seg_blueprint = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_blueprint.set_attribute('image_size_x', str(image_size_x))
        seg_blueprint.set_attribute('image_size_y', str(image_size_y))
        seg_blueprint.set_attribute('fov', str(fov))
        seg_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        seg_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        seg_sensor = self.world.spawn_actor(seg_blueprint, seg_transform, self.static_actor)
        if self.segmentation_sensor_active:
            seg_sensor.listen(lambda data: self.process_segmentation(data))

        self.semantic_segmentation_sensor = seg_sensor

        # 创建激光雷达传感器
        lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # ---------------------------------------------------------------
        # lidar_blueprint.set_attribute("channels", '128')
        # lidar_blueprint.set_attribute('range', '75.0')
        # lidar_blueprint.set_attribute('rotation_frequency', '100.0')
        # lidar_blueprint.set_attribute('horizontal_fov', '90.0')
        # lidar_blueprint.set_attribute('upper_fov', '45.0')
        # lidar_blueprint.set_attribute('lower_fov', '-45.0')
        # lidar_blueprint.set_attribute('points_per_second', '2500000')
        # lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))
        # # 设置激光雷达的变换
        # lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))

        # ---------------------------------------------------------------
        lidar_blueprint.set_attribute("channels", '256')
        lidar_blueprint.set_attribute('range', '100.0')
        lidar_blueprint.set_attribute('rotation_frequency', '100.0')
        lidar_blueprint.set_attribute('horizontal_fov', '360.0')
        lidar_blueprint.set_attribute('upper_fov', '0.0')
        lidar_blueprint.set_attribute('lower_fov', '-90.0')
        lidar_blueprint.set_attribute('points_per_second', '25000000')
        lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))
        # 设置激光雷达的变换
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=0))

        # -----------------------------------------------------------------
        lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_transform, self.static_actor)

        # 创建垂直向下的 RGB 相机
        rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        # 设置相机的变换（位置和旋转）
        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)

        # 监听传感器数据
        if self.dot_sensors_active:
            lidar_sensor.listen(lambda data: self.process_lidar(data))

        if self.rgb_sensors_active:
            rgb_sensor.listen(lambda data: self.process_rgb("camera4", data))
        
        self.lidar_sensor = lidar_sensor
        self.sensors.append(rgb_sensor)

        # 创建四个不同方向的 RGB 相机
        for direction, yaw, offset in zip(directions, yaw_angles, sensor_offset):
            adjusted_yaw = yaw # 根据无人机的朝向调整传感器的朝向

            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(
                carla.Location(x=offset[0], y=offset[1], z=offset[2]),
                carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree)
            )
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            if self.rgb_sensors_active:
                rgb_sensor.listen(lambda data, dir=direction: self.process_rgb(dir, data))
            self.sensors.append(rgb_sensor)

    def process_lidar(self, data):
        # 暂存LiDAR数据
        self.sensor_queue.put((data, "lidar"))
        if data.frame > UAV.max_frame:
            UAV.max_frame = data.frame
        self.sensors_data_counter += 1

    def process_rgb(self, sensor_id, data):
        # 暂存RGB数据
        self.sensor_queue.put((data, sensor_id))
        if data.frame > UAV.max_frame:
            UAV.max_frame = data.frame
        self.sensors_data_counter += 1

    def process_segmentation(self, data):
        # 暂存语义分割数据
        self.sensor_queue.put((data, "segmentation"))
        if data.frame > UAV.max_frame:
            UAV.max_frame = data.frame
        self.sensors_data_counter += 1


    def save_segmentation(self, data, frame):
        """
        处理并保存语义分割图像数据，基于 CARLA API 生成五种 BEV 语义分割图像。

        参数：
        - data：传感器返回的语义分割图像数据（实际不使用）。
        - frame：当前帧号。
        """
        # BEV 参数设置
        bev_size = 256
        bev_range = 51.2  # 从 -51.2 到 51.2
        resolution = (2 * bev_range) / bev_size  # 每个像素代表的实际距离

        # 获取 UAV 位置
        uav_location = self.static_actor.get_location()
        
        # 生成三种基础 BEV 图像
        bev_dynamic = self._generate_bev_dynamic(uav_location, bev_size, bev_range, resolution)
        bev_static = self._generate_bev_static(uav_location, bev_size, bev_range, resolution)
        bev_lane = self._generate_bev_lane(uav_location, bev_size, bev_range, resolution)

        # 生成 BEV visibility 图像（基于 self.lidar_data）
        bev_visibility = self._generate_bev_visibility(uav_location, bev_size, bev_range, resolution)
        bev_visibility_corp = np.copy(bev_visibility)  # 协同可见性图像直接复制

        # 保存图像（所有BEV图像逆时针旋转90度）
        Image.fromarray(np.rot90(bev_dynamic, k=1)).save(os.path.join(self.self_dir, f'{frame}_bev_dynamic.png'))
        Image.fromarray(np.rot90(bev_static, k=1)).save(os.path.join(self.self_dir, f'{frame}_bev_static.png'))
        Image.fromarray(np.rot90(bev_lane, k=1)).save(os.path.join(self.self_dir, f'{frame}_bev_lane.png'))
        Image.fromarray(np.rot90(bev_visibility, k=1)).save(os.path.join(self.self_dir, f'{frame}_bev_visibility.png'))
        Image.fromarray(np.rot90(bev_visibility_corp, k=1)).save(os.path.join(self.self_dir, f'{frame}_bev_visibility_corp.png'))

    def _world_to_bev_coords(self, world_x, world_y, uav_x, uav_y, bev_size, bev_range, uav_yaw=0):
        """将世界坐标转换为 BEV 像素坐标，考虑无人机朝向"""
        # 相对于 UAV 的坐标
        rel_x = world_x - uav_x
        rel_y = world_y - uav_y
        
        # 考虑无人机朝向的旋转变换
        yaw_rad = math.radians(uav_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        
        # 旋转坐标
        rotated_x = rel_x * cos_yaw + rel_y * sin_yaw
        rotated_y = -rel_x * sin_yaw + rel_y * cos_yaw
        
        # 转换为像素坐标
        pixel_x = int((rotated_x + bev_range) / (2 * bev_range) * bev_size)
        pixel_y = int((rotated_y + bev_range) / (2 * bev_range) * bev_size)
        
        return pixel_x, pixel_y

    def _generate_bev_dynamic(self, uav_location, bev_size, bev_range, resolution):
        """生成动态目标 BEV 图像"""
        bev_image = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 获取无人机朝向
        uav_yaw = self.static_actor.get_transform().rotation.yaw
        
        # 获取所有车辆
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicles:
            vehicle_location = vehicle.get_location()
            
            # 检查是否在 BEV 范围内
            rel_x = vehicle_location.x - uav_location.x
            rel_y = vehicle_location.y - uav_location.y
            
            if abs(rel_x) <= bev_range and abs(rel_y) <= bev_range:
                # 获取车辆边界框
                bbox = vehicle.bounding_box
                transform = vehicle.get_transform()
                
                # 计算车辆四个角点
                corners = self._get_vehicle_corners(bbox, transform)
                
                # 将角点转换为 BEV 坐标并绘制
                bev_corners = []
                for corner in corners:
                    pixel_x, pixel_y = self._world_to_bev_coords(
                        corner[0], corner[1], uav_location.x, uav_location.y, bev_size, bev_range, uav_yaw
                    )
                    if 0 <= pixel_x < bev_size and 0 <= pixel_y < bev_size:
                        bev_corners.append([pixel_x, pixel_y])
                
                if len(bev_corners) >= 3:
                    # 填充多边形
                    from PIL import ImageDraw
                    pil_image = Image.fromarray(bev_image)
                    draw = ImageDraw.Draw(pil_image)
                    draw.polygon([tuple(corner) for corner in bev_corners], fill=(255, 255, 255))
                    bev_image = np.array(pil_image)
        
        return bev_image

    def _generate_bev_static(self, uav_location, bev_size, bev_range, resolution):
        """生成静态可行驶区域 BEV 图像"""
        bev_image = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 获取无人机朝向
        uav_yaw = self.static_actor.get_transform().rotation.yaw
        
        # 获取地图
        world_map = self.world.get_map()
        
        # 遍历 BEV 范围内的每个像素
        for i in range(bev_size):
            for j in range(bev_size):
                # 将像素坐标转换回世界坐标（考虑朝向）
                rel_x = (i - bev_size/2) * resolution
                rel_y = (j - bev_size/2) * resolution
                
                # 反向旋转变换
                yaw_rad = math.radians(uav_yaw)
                cos_yaw = math.cos(-yaw_rad)
                sin_yaw = math.sin(-yaw_rad)
                
                world_rel_x = rel_x * cos_yaw + rel_y * sin_yaw
                world_rel_y = -rel_x * sin_yaw + rel_y * cos_yaw
                
                world_x = uav_location.x + world_rel_x
                world_y = uav_location.y + world_rel_y
                
                # 检查该点是否在道路上
                location = carla.Location(world_x, world_y, uav_location.z)
                waypoint = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
                
                if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                    # 检查距离阈值
                    distance = math.sqrt((waypoint.transform.location.x - world_x)**2 + 
                                       (waypoint.transform.location.y - world_y)**2)
                    if distance < 3.0:  # 3米阈值
                        bev_image[j, i] = [255, 255, 255]
        
        return bev_image

    def _generate_bev_lane(self, uav_location, bev_size, bev_range, resolution):
        """生成车道线 BEV 图像"""
        bev_image = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 获取无人机朝向
        uav_yaw = self.static_actor.get_transform().rotation.yaw
        
        # 获取地图
        world_map = self.world.get_map()
        
        # 减少采样间隔，使线条更连续
        for i in range(0, bev_size, 1):  # 从2改为1，使线条更密
            for j in range(0, bev_size, 1):
                # 将像素坐标转换回世界坐标（考虑朝向）
                rel_x = (i - bev_size/2) * resolution
                rel_y = (j - bev_size/2) * resolution
                
                # 反向旋转变换
                yaw_rad = math.radians(uav_yaw)
                cos_yaw = math.cos(-yaw_rad)
                sin_yaw = math.sin(-yaw_rad)
                
                world_rel_x = rel_x * cos_yaw + rel_y * sin_yaw
                world_rel_y = -rel_x * sin_yaw + rel_y * cos_yaw
                
                world_x = uav_location.x + world_rel_x
                world_y = uav_location.y + world_rel_y
                
                location = carla.Location(world_x, world_y, uav_location.z)
                waypoint = world_map.get_waypoint(location, project_to_road=True)
                
                if waypoint:
                    # 检查是否接近车道边界
                    lane_width = waypoint.lane_width
                    
                    # 获取左右边界点
                    left_lane_marking = waypoint.left_lane_marking
                    right_lane_marking = waypoint.right_lane_marking
                    
                    if (left_lane_marking.type != carla.LaneMarkingType.NONE or 
                        right_lane_marking.type != carla.LaneMarkingType.NONE):
                        
                        # 计算到车道中心的横向距离
                        waypoint_location = waypoint.transform.location
                        dx = world_x - waypoint_location.x
                        dy = world_y - waypoint_location.y
                        
                        # 计算垂直于车道的距离
                        yaw_wp = math.radians(waypoint.transform.rotation.yaw)
                        lateral_distance = abs(-dx * math.sin(yaw_wp) + dy * math.cos(yaw_wp))
                        
                        # 更宽松的车道线检测阈值
                        if abs(lateral_distance - lane_width/2) < 0.5:  # 从0.3改为0.5
                            bev_image[j, i] = [255, 255, 255]
        
        return bev_image

    def _generate_bev_visibility(self, uav_location, bev_size, bev_range, resolution):
        """生成 BEV 可见性图像，基于 self.lidar_data"""
        bev_image = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
        
        # 检查是否有 LiDAR 数据
        if self.lidar_data is None:
            print(f"UAV {self.uav_id}: No LiDAR data available for visibility generation")
            return bev_image
        
        # 获取无人机朝向
        uav_yaw = self.static_actor.get_transform().rotation.yaw
        
        # 获取所有车辆
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        # 处理 LiDAR 数据，转换为世界坐标
        lidar_data_raw = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
        lidar_data_raw = np.reshape(lidar_data_raw, (int(lidar_data_raw.shape[0] / 4), 4))
        lidar_points = lidar_data_raw[:, :3]
        world_points = self.local_to_world(lidar_points)
        
        detected_count = 0
        drawn_count = 0
        
        for vehicle in vehicles:
            vehicle_location = vehicle.get_location()
            
            # 第一步：检查是否在 BEV 范围内（和 _generate_bev_dynamic 完全一样）
            rel_x = vehicle_location.x - uav_location.x
            rel_y = vehicle_location.y - uav_location.y
            
            if abs(rel_x) <= bev_range and abs(rel_y) <= bev_range:
                # 第二步：检查是否被 LiDAR 检测到（使用 self.lidar_data 的点云数据）
                if self.vehicle_in_lidar(vehicle, world_points):
                    detected_count += 1
                    
                    # 第三步：绘制车辆（和 _generate_bev_dynamic 完全一样）
                    bbox = vehicle.bounding_box
                    transform = vehicle.get_transform()
                    
                    # 计算车辆四个角点
                    corners = self._get_vehicle_corners(bbox, transform)
                    
                    # 将角点转换为 BEV 坐标并绘制
                    bev_corners = []
                    for corner in corners:
                        pixel_x, pixel_y = self._world_to_bev_coords(
                            corner[0], corner[1], uav_location.x, uav_location.y, bev_size, bev_range, uav_yaw
                        )
                        if 0 <= pixel_x < bev_size and 0 <= pixel_y < bev_size:
                            bev_corners.append([pixel_x, pixel_y])
                    
                    if len(bev_corners) >= 3:
                        # 填充多边形
                        from PIL import ImageDraw
                        pil_image = Image.fromarray(bev_image)
                        draw = ImageDraw.Draw(pil_image)
                        draw.polygon([tuple(corner) for corner in bev_corners], fill=(255, 255, 255))
                        bev_image = np.array(pil_image)
                        drawn_count += 1
        
        print(f"UAV {self.uav_id}: Detected {detected_count} vehicles in visibility, drew {drawn_count} vehicles")
        return bev_image

    def _get_vehicle_corners(self, bbox, transform):
        """获取车辆边界框的四个角点世界坐标"""
        # 边界框的局部坐标
        corners_local = [
            [-bbox.extent.x, -bbox.extent.y, 0],
            [bbox.extent.x, -bbox.extent.y, 0],
            [bbox.extent.x, bbox.extent.y, 0],
            [-bbox.extent.x, bbox.extent.y, 0]
        ]
        
        corners_world = []
        for corner in corners_local:
            # 将局部坐标转换为世界坐标
            location = carla.Location(corner[0], corner[1], corner[2])
            world_location = transform.transform(location)
            corners_world.append([world_location.x, world_location.y])
        
        return corners_world


    def save_image(self, image, direction, frame):
        """
        处理并保存传感器的图像数据。

        参数：
        - image：传感器返回的图像数据。
        - direction：图像的方向标签。
        """
        # 生成文件名并保存图像
        if image != None:
            file_name = os.path.join(self.self_dir, f'{frame}_{direction}.png')
            # 解析 BGRA 数据并转换为 RGB
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            rgb_array = array[:, :, :3][:, :, ::-1]  # 去掉 Alpha 通道（BGRA → RGB）

            # 保存为 RGB 格式的 PNG
            img = Image.fromarray(rgb_array)
            img.save(file_name)

    def save_lidar(self, image, frame):
        """
        处理并保存激光雷达的点云数据。

        参数：
        - image：传感器返回的点云数据。
        - frame：当前帧号。
        """
        # 将原始数据转换为 numpy 数组
        if image != None:
            data = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype('f4')))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))

            # 提取 XYZ 坐标
            points = data[:, :3]

            points=self.local_to_world(points)
            points[:, 0] -= self.lidar_sensor.get_location().x
            points[:, 1] -= self.lidar_sensor.get_location().y
            # points[:, 2] += self.lidar_sensor.get_location().z

            # 提取 intensity 信息
            intensities = data[:, 3]

            # 标准化 intensity 到 [0, 1] 范围
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            intensities = np.clip(intensities, 0, 1)

            # 使用 matplotlib colormap 将 intensity 映射为彩色
            colormap = plt.get_cmap('viridis')
            colors = colormap(intensities)[:, :3]  # 仅提取 RGB 通道

            # 创建点云并保存为 PCD 文件
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd_file_name = os.path.join(self.self_dir, f'{frame}.pcd')
            o3d.io.write_point_cloud(pcd_file_name, pcd)

    def get_intrinsics(self):
        """
        计算传感器的内参矩阵。

        参数：
        - sensor：传感器对象。

        返回：
        - intrinsics：3x3 的 numpy 数组，表示内参矩阵。
        """
        sensor = self.sensors[0]

        fov = float(sensor.attributes['fov'])
        image_width = int(sensor.attributes['image_size_x'])
        image_height = int(sensor.attributes['image_size_y'])

        # 将 FOV 从度转换为弧度
        fov_rad = fov * np.pi / 180.0  # 水平视场角（弧度）

        # 计算垂直视场角
        aspect_ratio = image_height / image_width
        vertical_fov_rad = 2.0 * np.arctan(np.tan(fov_rad / 2.0) * aspect_ratio)

        # 计算焦距
        fx = image_width / (2.0 * np.tan(fov_rad / 2.0))
        fy = image_height / (2.0 * np.tan(vertical_fov_rad / 2.0))

        # 构造内参矩阵
        intrinsics = np.array([
            [fx, 0, image_width / 2],
            [0, fy, image_height / 2],
            [0, 0, 1]
        ])
        
        return intrinsics


    def get_sensor_extrinsics_and_pose(self, data):
        """
        计算传感器相对于给定坐标系的外参矩阵和位姿。

        参数：
        - data：包含传感器位姿信息的对象。

        返回：
        - extrinsics：4x4 的 numpy 数组，表示外参矩阵。
        - pose：长度为 6 的 numpy 数组，包含 [x, y, z, roll, pitch, yaw]，角度以度为单位。
        """
        sensor_transform = data.transform

        # 提取位置信息
        x = sensor_transform.location.x
        y = sensor_transform.location.y
        z = sensor_transform.location.z

        # 提取旋转信息（Roll、Pitch、Yaw 以度为单位）
        roll_deg = sensor_transform.rotation.roll
        yaw_deg = sensor_transform.rotation.yaw
        pitch_deg = sensor_transform.rotation.pitch

        # 保存位姿
        pose = np.array([x, y, z, roll_deg, yaw_deg, pitch_deg])

        # 将欧拉角直接转换为旋转矩阵（使用四元数避免万向节锁）
        # 使用 CARLA 提供的函数获取旋转矩阵
        t_sensor_to_world = np.array(sensor_transform.get_matrix())  # 获取齐次变换矩阵
        rotation_matrix = t_sensor_to_world[:3, :3]  # 提取传感器到世界的旋转矩阵

        # 计算从世界到传感器的旋转矩阵
        r_world_to_sensor = rotation_matrix.T  # 旋转矩阵的转置

        # 定义 CARLA 坐标系到相机坐标系的转换矩阵
        r_carla_to_camera = np.array([
            [0, -1,  0],  # Y -> -X
            [0,  0, -1],  # Z -> -Y
            [1,  0,  0]   # X -> Z
        ])

        # 将 CARLA 坐标系旋转转换为相机坐标系
        R = r_carla_to_camera @ r_world_to_sensor

        # 构建 4x4 外参矩阵
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = - R @ np.array([x, y, z])

        return extrinsics, pose
  
    def get_lidar_pose(self, data):
        sensor_transform = data.transform

        x = sensor_transform.location.x
        y = sensor_transform.location.y
        # z = sensor_transform.location.z

        # roll = sensor_transform.rotation.roll
        # yaw = sensor_transform.rotation.yaw
        # pitch = sensor_transform.rotation.pitch

        # pose = np.array([x, y, 0, roll, yaw, pitch])
        pose = np.array([x, y, 0, 0, 0, 0]) # 激光雷达的高度设为0，并且是水平放置的

        return pose

    def check_and_save_all(self, vehicles):
        """
        保存参数到 YAML 文件。
        """
        # 创建一个字典用于存储所有相机参数
        camera_params = {}

        yaml_file = None

        if self.sensors_data_counter < self.total_sensors:
            return

        try:
            # 先处理 lidar 数据，确保 self.lidar_data 可用
            lidar_data_processed = False
            segmentation_data = None
            other_data = []

            # 收集所有数据并按类型分组
            for _ in range(7):
                data = self.sensor_queue.get(True, 1.0)

                if data is None:
                    continue

                if data[1] == "lidar":
                    # 优先处理 lidar 数据
                    yaml_file = os.path.join(self.self_dir, f'{UAV.max_frame}.yaml')
                    self.lidar_data = data[0]  # 保存 LiDAR 数据供其他方法使用
                    lidar_pose = self.get_lidar_pose(data[0])
                    camera_params['lidar_pose'] = lidar_pose.tolist()
                    self.save_lidar(data[0], UAV.max_frame)
                    lidar_data_processed = True
                elif data[1] == "segmentation":
                    segmentation_data = data
                else:
                    other_data.append(data)

            # 处理 segmentation 数据（在 lidar 之后，现在包含 visibility 生成）
            if segmentation_data is not None:
                self.save_segmentation(segmentation_data[0], UAV.max_frame)

            # 处理其他传感器数据
            for data in other_data:
                camera_id = data[1]
                # 获取外参和位姿
                extrinsics, pose = self.get_sensor_extrinsics_and_pose(data[0])
                intrinsics = self.get_intrinsics()

                # 将相机的参数添加到字典中
                camera_params[camera_id] = {
                    'cords': pose.tolist(),
                    'extrinsic': extrinsics.tolist(),
                    'intrinsic': intrinsics.tolist()
                }

                # 保存相机图像
                self.save_image(data[0], camera_id, UAV.max_frame)
                
            # 补全OPENCOOD的数据格式
            # camera_params['ego_speed'] = 0
            # camera_params['plan_trajectory'] = [[0,0,0]]
            # camera_params['predicted_ego_pos'] = [0,0,0,0,0,0]
            # camera_params['true_ego_pos'] = [0,0,0,0,0,0]

            # 将所有相机的参数写入一个 YAML 文件
            if yaml_file is not None:
                self.save_camera_params_to_yaml(camera_params, yaml_file)
                self.add_vehicle_to_yaml(vehicles, self.lidar_data, yaml_file)
                self.sensors_data_counter = 0
        except Empty:
            print("Queue is empty")

    def save_camera_params_to_yaml(self, camera_params, yaml_file):
        """
        将相机参数保存到 YAML 文件。

        参数：
        - camera_params：包含相机参数的字典。
        - yaml_file：YAML 文件的路径。
        """
        # 写入文件
        try:
            with open(yaml_file, 'w') as file:
                yaml.dump(camera_params, file, default_flow_style=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    def add_vehicle_to_yaml(self, vehicles_list, lidar_data, yaml_file):

        vehicles_info = {}
        count=0

        # 将原始点云数据转换为 numpy 数组
        lidar_points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]

        # print(lidar_points[0])
        world_points = self.local_to_world(lidar_points)

        lidar_location = self.lidar_sensor.get_location()
        lidar_position =[lidar_location.x,lidar_location.y]

        def euclidean_distance(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


        for vehicle in vehicles_list:
            vehicle_position = vehicle.get_location()  # 车辆位置，假设为一个坐标对象 [x, y, z]
            distance = euclidean_distance(lidar_position, [vehicle_position.x, vehicle_position.y])
            if distance > 150:
                continue
            if(self.vehicle_in_lidar(vehicle,world_points)):
                # print(vehicle.get_location())
                # print(vehicle.type_id)
                # print("--------------------")
                # # 收集车辆信息
                # location = vehicle.get_location()
                # rotation = vehicle.get_transform().rotation
                velocity = vehicle.get_velocity()  # 获取车辆的线速度
                speed = 3.6 * ((velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)  # m/s 转 km/h

                vehicle_info = {

                    'angle': [
                        vehicle.get_transform().rotation.roll,
                        vehicle.get_transform().rotation.yaw,
                        vehicle.get_transform().rotation.pitch
                    ],
                    'center': [
                        vehicle.bounding_box.location.x,
                        vehicle.bounding_box.location.y,
                        vehicle.bounding_box.location.z
                    ],
                    'extent': [
                        vehicle.bounding_box.extent.x,
                        vehicle.bounding_box.extent.y,
                        vehicle.bounding_box.extent.z
                    ],
                    'location': [
                        vehicle.get_location().x,
                        vehicle.get_location().y,
                        vehicle.get_location().z
                    ],
                    'speed': speed  # 计算速度
                }

                # 将信息存入字典
                vehicles_info[vehicle.id] = vehicle_info
                count=count+1
        print("%d:%d"%(self.uav_id,count))


        # 将车辆信息存入 YAML 文件
        with open(yaml_file, 'r', encoding='utf-8') as file:
            # 读取现有数据
            existing_data = yaml.safe_load(file)
        existing_data['vehicles'] = vehicles_info

        with open(yaml_file, 'w', encoding='utf-8') as file:
            yaml.dump(existing_data, file, allow_unicode=True, default_flow_style=False)

        # print("车辆信息已保存到 %s" % yaml_file)

    def vehicle_in_lidar_enhanced(self, vehicle, world_points):
        """
        改进的车辆LiDAR检测函数，使用更宽松的检测标准
        :param vehicle: 车辆对象
        :param world_points: LiDAR 传感器获取的点云数据
        :return: 如果车辆被扫到则返回 True，否则返回 False
        """
        vehicle_location = vehicle.get_location()
        vehicle_bbox = vehicle.bounding_box
        
        # 车辆中心位置
        vehicle_center = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z])
        
        # 使用车辆边界框的范围作为检测区域
        bbox_extent = max(vehicle_bbox.extent.x, vehicle_bbox.extent.y)
        detection_radius = bbox_extent + 1.0  # 增加1米的缓冲区域
        
        # 计算点云到车辆中心的距离
        distances = np.linalg.norm(world_points - vehicle_center, axis=1)
        
        # 检查是否有点云在车辆附近
        nearby_points = world_points[distances < detection_radius]
        
        # 更严格的检查：确保有足够的点云在车辆高度范围内
        if len(nearby_points) > 0:
            height_filtered = nearby_points[
                (nearby_points[:, 2] >= vehicle_center[2] - 1.0) & 
                (nearby_points[:, 2] <= vehicle_center[2] + 2.0)
            ]
            return len(height_filtered) > 2  # 至少需要3个点
        
        return False

    def vehicle_in_lidar(self,vehicle, world_points):
        """
        判断车辆是否被 LiDAR 传感器扫描到
        :param vehicle: 车辆对象
        :param lidar_points: LiDAR 传感器获取的点云数据
        :return: 如果车辆被扫到则返回 True，否则返回 False
        """

        # vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle.get_location()

        vehicle_pos = np.array([vehicle_location.x, vehicle_location.y, vehicle_location.z])

        # 根据车辆位置过滤世界坐标系下的点云数据
        distances = np.linalg.norm(world_points - vehicle_pos, axis=1)
        detected_points = world_points[distances < 2]  # 阈值可以根据需要调整

        # 若检测到车辆，则打印车辆信息
        if len(detected_points) > 0:
            return True
        else:
            return False
            # print(f"Vehicle ID: {vehicle.id}, Location: {vehicle_location}")

    def local_to_world(self,points):
        """
        将局部点云坐标转换为世界坐标系
        :param points: N x 3 的点云数组
        :param sensor_transform: carla.Transform 对象
        :return: N x 3 的世界坐标点云
        """

        sensor_transform=self.lidar_sensor.get_transform()


        num_points = points.shape[0]
        points_homogeneous = np.hstack((points, np.ones((num_points, 1))))  # N x 4

        # 获取 4x4 齐次变换矩阵
        transform_matrix = np.array(sensor_transform.get_matrix())  # 4x4

        # 直接左乘矩阵
        transformed_points_homogeneous = np.dot(points_homogeneous, transform_matrix.T)  # N x 4

        # 转换回 3D 坐标
        transformed_points = transformed_points_homogeneous[:, :3]  # 取前三列


        return transformed_points


    def move(self):
        """
        控制无人机移动，并添加随机扰动。
        只有在移动开关打开时才会移动。
        """
        # 生成移动的随机扰动
        noise = np.random.normal(0, self.noise_std, size=3)

        delta_location_with_noise = carla.Location(
            x=self.delta_location.x + noise[0],
            y=self.delta_location.y + noise[1],
            z=self.delta_location.z + noise[2]
        )

        # 更新无人机的位置
        new_location = self.static_actor.get_location() + delta_location_with_noise
        self.static_actor.set_location(new_location)

    def update(self,vehicles, current_tick):
        """
        每次 tick 调用该方法来检查是否需要移动无人机。

        每次 tick 会统一处理并写入所有传感器的数据。
        """
        self.check_and_save_all(vehicles)


        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_capture:
                self.tick_counter = 0
                self.move()

    # 设置各种参数的函数
    def set_root_dir(self, new_dir):
        self.root_dir = new_dir

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
        """
        清理资源，销毁传感器和静态演员。
        """
        # time.sleep(1)

        # 销毁所有传感器
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        # 销毁静态演员
        if self.static_actor is not None and self.static_actor.is_alive:
            self.static_actor.destroy()
