#!/usr/bin/env python

import sys
import os
import time
import random  # 导入标准库的 random 模块

import carla
import numpy as np

from queue import Queue
from queue import Empty
import argparse

from uav_utils import UAV  # 导入自定义的 UAV 类

# 替换为你的 agents 所在目录
agents_path = r"C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\PythonAPI\carla"
if agents_path not in sys.path:
    sys.path.append(agents_path)


# To import a basic agent
from agents.navigation.basic_agent import BasicAgent

# To import a behavior agent
from agents.navigation.behavior_agent import BehaviorAgent


def get_actor_blueprints(world, filter_pattern):
    """
    获取与给定过滤器模式匹配的演员蓝图。

    参数：
    - world：CARLA 世界对象
    - filter_pattern：蓝图过滤器模式，例如 'vehicle.*'
    """
    return world.get_blueprint_library().filter(filter_pattern)


def main(world_name, simulation_sec, save_Dir):
    # 存储生成的车辆演员 ID，以便后续销毁
    vehicles_list = []

    # 设置 CARLA 客户端和世界
    host = '127.0.0.1'
    port = 2000
    synchronous_mode = True  # 是否使用同步模式
    tm_port = 8000  # 交通管理器端口

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    # 加载特定的地图
    world = client.load_world_if_different(world_name)
    if not world:
        world = client.get_world()

    # 获取交通管理器和世界设置
    traffic_manager = client.get_trafficmanager(tm_port)
    settings = world.get_settings()

    # 保存默认设置，以便后续恢复
    default_substep = settings.max_substeps
    default_substep_delta_time = settings.max_substep_delta_time

    print("Default max_substeps:", default_substep)
    print("Default max_substep_delta_time:", default_substep_delta_time)

    # 设置同步模式和物理子步参数
    if synchronous_mode:
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.max_substep_delta_time = 0.001
        settings.max_substeps = 10  # 有效范围：1-16
        settings.fixed_delta_seconds = 0.01  # 应小于 0.5
        world.apply_settings(settings)

    total_sec = simulation_sec  # 模拟总时长（秒）
    total_tick = int(total_sec / settings.fixed_delta_seconds) + 1  # 总 tick 数

    try:
        # ----------------- 天气设置 -----------------
        # 预定义的天气参数
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

        # 自定义天气参数（当前未使用）
        custom_weather = carla.WeatherParameters(
            cloudiness=80.0,             # 云量
            precipitation=30.0,          # 降水量
            precipitation_deposits=30.0, # 降水积累
            wind_intensity=10.0,         # 风力强度
            sun_azimuth_angle=90.0,      # 太阳方位角
            sun_altitude_angle=45.0      # 太阳高度角
        )

        # 选择并设置天气预设
        selected_weather = "ClearSunset"
        world.set_weather(weather_presets[selected_weather])

        # ----------------- 地图设置 -----------------
        # 获取所有路口
        junctions = world.get_map().get_topology()
        junction_locations = [junction[0].transform.location for junction in junctions]

        # 打印所有路口的位置
        # for i, location in enumerate(junction_locations):
        #     print(f"Junction {i}: (x={location.x}, y={location.y}, z={location.z})")

        # ----------------- 交通设置 -----------------

        desired_vehicle_number = 100  # 想要生成的车辆数量

        # 获取车辆蓝图
        blueprints = get_actor_blueprints(world, 'vehicle.*')
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")

        # 获取可用的出生点
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # 打印出生点数量和期望的车辆数量
        print("Number of spawn points:", number_of_spawn_points)
        print("Desired number of vehicles:", desired_vehicle_number)

        if desired_vehicle_number > number_of_spawn_points:
            desired_vehicle_number = number_of_spawn_points

        # 随机打乱出生点列表
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

            # 创建生成车辆的命令并设置自动驾驶
            spawn_actor = carla.command.SpawnActor(blueprint, transform)
            batch.append(spawn_actor)

        # 批量生成车辆，并分配agent
        agents = []
        responses = client.apply_batch_sync(batch, synchronous_mode)
        for response in responses:
            if response.error:
                print(response.error)
            else:
                vehicle = world.get_actor(response.actor_id)
                vehicles_list.append(vehicle)

                agent = BasicAgent(vehicle)
                agents.append(agent)

                destination = random.choice(spawn_points).location
                agent.set_destination(destination)

                agent.set_target_speed(100)


        # ----------------- UAV 设置 -----------------
        uavs = []

        # 创建第一个 UAV
        # 随机选择一个路口位置作为第一个 UAV 的位置
        # location1 = random.choice(junction_locations)
        # location1.z += 50  # 提升 UAV 的高度

        location1 = carla.Location(x=0, y=0, z=50)
        uav1 = UAV(world, location1, uav_id=1, rootDir=save_Dir, yaw_angle=random.uniform(0, 360))
        uavs.append(uav1)

        # # 随机生成4个 UAV
        # for i in range(2, 6):
        #     angle = random.uniform(0, 2 * np.pi)
        #     radius = random.uniform(0, 50)
        #     x_offset = radius * np.cos(angle)
        #     y_offset = radius * np.sin(angle)
        #     location = carla.Location(x=location1.x + x_offset, y=location1.y + y_offset, z=location1.z)
        #     yaw_angle = random.uniform(0, 360)
        #     uav = UAV(world, location, uav_id=i, rootDir=save_Dir, yaw_angle=yaw_angle)
        #     uavs.append(uav)



        # ----------------- 开始模拟 -----------------
        tick_count = 0

        while True:

            if synchronous_mode:
                world.tick()
                # print(f"Tick {tick_count}/{total_tick}")
                tick_count += 1

                for i, agent in enumerate(agents):
                    if agent.done():
                        destination = random.choice(spawn_points).location
                        agent.set_destination(destination)
                    
                    control = agent.run_step()
                    vehicles_list[i].apply_control(control)

                for uav in uavs:
                    vehicles = world.get_actors().filter('vehicle.*')
                    uav.update(vehicles)
            else:
                world.wait_for_tick()

            if tick_count > total_tick:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # ----------------- 资源清理 -----------------
        # 恢复世界设置
        if synchronous_mode and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.max_substep_delta_time = default_substep_delta_time
            settings.max_substeps = default_substep
            world.apply_settings(settings)

        # 销毁车辆演员
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # 销毁 UAV
        for uav in uavs:
            uav.destroy()


def parse_arguments():
    parser = argparse.ArgumentParser(description='CARLA Auto Drive with Sensors')
    parser.add_argument('--repetitions', type=int, required=True, help='Number of repetitions for data collection')
    parser.add_argument('--town', type=str, required=True, help='Name of the town map to use')
    return parser.parse_args()

args = parse_arguments()

if __name__ == '__main__':
    try:
        base_dir = fr'C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\myDemo\dataset\{args.town.lower()}'
        for i in range(args.repetitions):
            save_dir = os.path.join(base_dir, f"run_{i+1}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"Running iteration {i+1} on {args.town}")
            main(args.town, 5, save_dir)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')