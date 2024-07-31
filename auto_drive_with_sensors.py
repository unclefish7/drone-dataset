#!/usr/bin/env python

import glob
import os
import sys
import time
import carla
import numpy as np
from numpy import random

vehicles_list = []
sensor_list = []
static_actor_list = []

# 设置CARLA客户端连接参数
host = '127.0.0.1'
port = 2000
synchronous_mode = True
tm_port = 8000

client = carla.Client(host, port)
client.set_timeout(10.0)

world = client.get_world()

traffic_manager = client.get_trafficmanager(tm_port)

settings = world.get_settings()
if synchronous_mode:
    traffic_manager.set_synchronous_mode(True)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 2
world.apply_settings(settings)

# 以上为基础设置
#####################################################################
def get_actor_blueprints(world, filter):
    return world.get_blueprint_library().filter(filter)

def process_image(image, direction, sensor_type):
    # Process and save image data with direction and sensor type in the filename
    image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

def process_depth_image(image, direction, sensor_type):
    # Process and save depth image data with direction and sensor type in the filename
    image.convert(carla.ColorConverter.LogarithmicDepth)
    image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    # lidar_data = depth_to_point_cloud(image)
    # np.savetxt(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_lidar_out\lidar_%s_%s_%06d.txt' % (direction, sensor_type, image.frame), lidar_data)

def depth_to_point_cloud(depth_image):
    # Convert depth image to point cloud
    # Get camera intrinsic parameters
    image_size_x = depth_image.width
    image_size_y = depth_image.height
    fov = float(depth_image.fov)
    focal_length = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = image_size_x / 2.0
    cy = image_size_y / 2.0

    depth_data = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
    depth_data = depth_data.reshape((depth_image.height, depth_image.width, 4))
    depth_data = depth_data[:, :, :3]

    points = []
    for v in range(image_size_y):
        for u in range(image_size_x):
            z = depth_data[v, u, 0] / 255.0 * 1000.0
            if z > 0.1:  # Ignore points that are too close
                x = (u - cx) * z / focal_length
                y = (v - cy) * z / focal_length
                points.append([x, y, z])

    return np.array(points)

def spawn_uav(image_size_x, image_size_y, fov, capture_intervals, pitch_degree):
    directions = ["North", "East", "South", "West"]
    yaw_angles = [0, 90, 180, 270]

    static_blueprint = world.get_blueprint_library().find('static.prop.box01')
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(yaw=0))
    static_actor = world.spawn_actor(static_blueprint, spawn_point)
    static_actor_list.append(static_actor)

    # 创建垂直向下的传感器
    # 创建RGB传感器并设置固定位置
    rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
    rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
    rgb_blueprint.set_attribute('fov', str(fov))
    rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

    rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
    rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
    rgb_sensor.listen(lambda data: process_image(data, "down", "rgb"))
    sensor_list.append(rgb_sensor)

    # 创建深度传感器并设置固定位置
    depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', str(image_size_x))
    depth_blueprint.set_attribute('image_size_y', str(image_size_y))
    depth_blueprint.set_attribute('fov', str(fov))
    depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

    depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
    depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
    depth_sensor.listen(lambda data: process_depth_image(data, "down", "depth"))
    sensor_list.append(depth_sensor)
    
    # 创建四个不同方向的传感器
    for direction, yaw in zip(directions, yaw_angles):
        # 创建RGB传感器并设置固定位置
        rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
        rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
        rgb_sensor.listen(lambda data, dir=direction: process_image(data, dir, "rgb"))
        sensor_list.append(rgb_sensor)

        # 创建深度传感器并设置固定位置
        depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        depth_blueprint.set_attribute('image_size_x', str(image_size_x))
        depth_blueprint.set_attribute('image_size_y', str(image_size_y))
        depth_blueprint.set_attribute('fov', str(fov))
        depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
        depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
        depth_sensor.listen(lambda data, dir=direction: process_depth_image(data, dir, "depth"))
        sensor_list.append(depth_sensor)

# 再深入了解一下转换的原理

def main():
    try:
####################################################################################
        # 以下为天气设置

        # 预定义的一些天气
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
            "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
            "MidRainSunset": carla.WeatherParameters.MidRainSunset,
            "HardRainSunset": carla.WeatherParameters.HardRainSunset,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
        }

        # 自定义天气
        custom_weather = carla.WeatherParameters(
            cloudiness=80.0,  # 云量
            precipitation=30.0,  # 降水量
            precipitation_deposits=30.0,  # 降水积累
            wind_intensity=10.0,  # 风力强度
            sun_azimuth_angle=90.0,  # 太阳方位角
            sun_altitude_angle=45.0  # 太阳高度角
        )

        # 选择一种天气设置
        selected_weather = "ClearNoon"
        world.set_weather(weather_presets[selected_weather])

####################################################################################
        # 以下是车流设置
        # 设置车辆间距
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        # 设置车辆速度
        traffic_manager.global_percentage_speed_difference(-20)
        # 设置是否重新生成静止车辆
        traffic_manager.set_respawn_dormant_vehicles(1)

        desired_vehicle_number = 100 #想要生成的车辆数目（不建议太多）

        blueprints = get_actor_blueprints(world, 'vehicle.*')
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # 查看一下spawn points和vehicles的数量
        print("number of spawn points:", number_of_spawn_points)
        print("desired number of vehicles:", desired_vehicle_number)

        if desired_vehicle_number > number_of_spawn_points:
            desired_vehicle_number = number_of_spawn_points

        random.shuffle(spawn_points)
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= desired_vehicle_number:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_mode):
            if response.error:
                print(response.error)
            else:
                vehicles_list.append(response.actor_id)

#################################################################################
        # 以下为传感器设置
        # 相关参数设置
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45
        fov = 90
        capture_intervals = 5.0

        # for i in range(5):
        #     spawn_uav(image_size_x=image_size_x, image_size_y=image_size_y, fov=fov, capture_intervals=capture_intervals, pitch_degree=pitch_degree)


        directions = ["North", "East", "South", "West"]
        yaw_angles = [0, 90, 180, 270]

        static_blueprint = world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(yaw=0))
        static_actor = world.spawn_actor(static_blueprint, spawn_point)
        static_actor_list.append(static_actor)

        # 创建垂直向下的传感器
        # 创建RGB传感器并设置固定位置
        rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
        rgb_sensor.listen(lambda data: process_image(data, "down", "rgb"))
        sensor_list.append(rgb_sensor)

        # 创建深度传感器并设置固定位置
        depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        depth_blueprint.set_attribute('image_size_x', str(image_size_x))
        depth_blueprint.set_attribute('image_size_y', str(image_size_y))
        depth_blueprint.set_attribute('fov', str(fov))
        depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
        depth_sensor.listen(lambda data: process_depth_image(data, "down", "depth"))
        sensor_list.append(depth_sensor)
        
        # 创建四个不同方向的传感器
        for direction, yaw in zip(directions, yaw_angles):
            # 创建RGB传感器并设置固定位置
            rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
            rgb_sensor.listen(lambda data, dir=direction: process_image(data, dir, "rgb"))
            sensor_list.append(rgb_sensor)

            # 创建深度传感器并设置固定位置
            depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', str(image_size_x))
            depth_blueprint.set_attribute('image_size_y', str(image_size_y))
            depth_blueprint.set_attribute('fov', str(fov))
            depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
            depth_sensor.listen(lambda data, dir=direction: process_depth_image(data, dir, "depth"))
            sensor_list.append(depth_sensor)

        # 参考co-perception学习一下传感器相关设置
        # 传感器位深、分辨率、视野fov、如何将深度图转换为点云
        # 让无人机进行移动
        # 让传感器有物理距离
        # 明确一下无人机的内外参
        # 采集信息的时候要包括无人机的内外参
        # 再学习一下数据的命名格式





#################################################################################
        # 开始运行
        while True:
            if synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick()

    finally:# 结束运行
        if synchronous_mode and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in sensor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in static_actor_list])
        time.sleep(0.5)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

# 无人机的路径规划可以添加随机扰动（高斯噪声），以追求现实不稳定的飞行状况