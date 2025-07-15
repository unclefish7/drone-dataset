#!/usr/bin/env python

import os
import time
import random  # 导入标准库的 random 模块

import carla
import numpy as np

from queue import Queue
from queue import Empty
import argparse

from uav_utils import UAV  # 导入自定义的 UAV 类


def get_actor_blueprints(world, filter_pattern):
    """
    获取与给定过滤器模式匹配的演员蓝图。

    参数：
    - world：CARLA 世界对象
    - filter_pattern：蓝图过滤器模式，例如 'vehicle.*'
    """
    return world.get_blueprint_library().filter(filter_pattern)


def main(world_name, simulation_sec, save_dir, locations, random_seed=0):

    random.seed(random_seed)

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
        # custom_weather = carla.WeatherParameters(
        #     cloudiness=80.0,             # 云量
        #     precipitation=30.0,          # 降水量
        #     precipitation_deposits=30.0, # 降水积累
        #     wind_intensity=10.0,         # 风力强度
        #     sun_azimuth_angle=90.0,      # 太阳方位角
        #     sun_altitude_angle=45.0      # 太阳高度角
        # )

        # 选择并设置天气预设
        selected_weather = "ClearSunset"
        world.set_weather(weather_presets[selected_weather])

        # ----------------- 地图设置 -----------------
        # 获取所有路口
        # junctions = world.get_map().get_topology()
        # junction_locations = [junction[0].transform.location for junction in junctions]

        # 打印所有路口的位置
        # for i, location in enumerate(junction_locations):
        #     print(f"Junction {i}: (x={location.x}, y={location.y}, z={location.z})")

        # ----------------- 交通设置 -----------------

        desired_vehicle_number = 150  # 想要生成的车辆数量

        # 获取车辆蓝图，仅选择中大型车辆
        blueprints = get_actor_blueprints(world, 'vehicle.*')
        # blueprints = [bp for bp in blueprints if not bp.id.startswith((
        #     'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets',
        #     'vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja', 'vehicle.vespa.zx125', 'vehicle.yamaha.yzf',
        #     'vehicle.micro.microlino', 'vehicle.mini.cooper_s', 'vehicle.mini.cooper_s_2021', 'vehicle.nissan.micra'
        # ))]
        blueprints = [bp for bp in blueprints if bp.id == 'vehicle.audi.a2']
        if not blueprints:
            raise ValueError("Couldn't find any suitable vehicles with the specified filters")

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
            random.seed(n+1)
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')

            # 创建生成车辆的命令并设置自动驾驶
            spawn_actor = carla.command.SpawnActor(blueprint, transform)
            set_autopilot = carla.command.SetAutopilot(
                carla.command.FutureActor, True, traffic_manager.get_port())
            batch.append(spawn_actor.then(set_autopilot))

        # 批量生成车辆
        responses = client.apply_batch_sync(batch, synchronous_mode)
        for response in responses:
            if response.error:
                print(response.error)
            else:
                vehicles_list.append(response.actor_id)

        traffic_manager.set_global_distance_to_leading_vehicle(0)  # 设置车辆间距
        traffic_manager.global_percentage_speed_difference(-99)      # 设置全局速度差异
        traffic_manager.set_respawn_dormant_vehicles(True)           # 重新生成静止车辆

        # ----------------- UAV 设置 -----------------
        uavs = []
        for i, location in enumerate(locations):
            uav = UAV(world, location, uav_id=i+1, root_dir=save_dir, yaw_angle=0)
            uavs.append(uav)

        # ----------------- 开始模拟 -----------------
        tick_count = 0

        # random.seed(1)

        while True:

            if synchronous_mode:
                world.tick()
                print(f"Tick {tick_count}/{total_tick}")
                tick_count += 1
                for uav in uavs:
                    vehicles = world.get_actors().filter('vehicle.*')
                    uav.update(vehicles,tick_count)
            else:
                world.wait_for_tick()

            if tick_count > total_tick:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # ----------------- 资源清理 -----------------
        # time.sleep(5)
        # 恢复世界设置
        if synchronous_mode and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.max_substep_delta_time = default_substep_delta_time
            settings.max_substeps = default_substep
            world.apply_settings(settings)

        # 销毁车辆演员
        # time.sleep(5)
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # 销毁 UAV
        for uav in uavs:
            uav.destroy()
            # time.sleep(2)
        # time.sleep(5)


def parse_arguments():
    parser = argparse.ArgumentParser(description='CARLA Auto Drive with Sensors')
    parser.add_argument('--town', type=str, required=True, help='Name of the town map to use')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for the simulation')
    return parser.parse_args()

args = parse_arguments()

if __name__ == '__main__':
    try:
        base_dir = r'E:\datasets\mydataset_OPV2V_seg_cam_lidar\train'
        # base_dir = fr'D:\CARLA_Latest\WindowsNoEditor\myDemo\dataset'

        test_points = {
            ###################################################
            # Town05, 15m
            'Town05_h15_1': [
                carla.Location(x=-122, y=-44, z=15),
                carla.Location(x=-157, y=-88, z=15),
                carla.Location(x=-189, y=-43, z=15),
                carla.Location(x=-155, y=1, z=15)
            ],
            'Town05_h15_2': [
                carla.Location(x=-50, y=0, z=15),
                carla.Location(x=-100, y=0, z=15),
                carla.Location(x=-20, y=0, z=15),
                carla.Location(x=-50, y=45, z=15)
            ],
            'Town05_h15_3': [
                carla.Location(x=-125, y=150, z=15),
                carla.Location(x=-125, y=90, z=15),
                carla.Location(x=-45, y=105, z=15),
                carla.Location(x=-190, y=125, z=15)
            ],
            ###################################################
            # Town03, 15m
            'Town03_h15_1': [
                carla.Location(x=43, y=-134, z=15),
                carla.Location(x=0, y=-85, z=15),
                carla.Location(x=-45, y=0, z=15),
                carla.Location(x=80, y=-85, z=15)
            ],
            'Town03_h15_2': [
                carla.Location(x=0, y=0, z=15),
                carla.Location(x=0, y=-50, z=15),
                carla.Location(x=0, y=50, z=15),
                carla.Location(x=50, y=0, z=15)
            ],
            'Town03_h15_3': [
                carla.Location(x=0, y=133, z=15),
                carla.Location(x=0, y=183, z=15),
                carla.Location(x=0, y=83, z=15),
                carla.Location(x=50, y=133, z=15)
            ],
            'Town03_h15_4': [
                carla.Location(x=-85, y=0, z=15),
                carla.Location(x=-40, y=0, z=15),
                carla.Location(x=-85, y=-50, z=15),
                carla.Location(x=-126, y=14, z=15)
            ],
            'Town03_h15_5': [
                carla.Location(x=171, y=60, z=15),
                carla.Location(x=150, y=130, z=15),
                carla.Location(x=235, y=-60, z=15),
                carla.Location(x=115, y=60, z=15)
            ],
            ###################################################
            # Town03, 5m
            "Town03_h05_1": [
                carla.Location(x=-6, y=22, z=5),
                carla.Location(x=23, y=-4, z=5),
                carla.Location(x=0, y=-22, z=5),
                carla.Location(x=-36, y=-1, z=5)
            ],
            "Town03_h05_2": [
                carla.Location(x=-80, y=-30, z=5),
                carla.Location(x=23, y=-4, z=5),
                carla.Location(x=0, y=-22, z=5),
                carla.Location(x=-36, y=-1, z=5)
            ],
            ###################################################
            # Town05, 50m
            'Town05_h50_1': [
                carla.Location(x=-122, y=-44, z=50),
                carla.Location(x=-157, y=-88, z=50),
                carla.Location(x=-189, y=-43, z=50),
                carla.Location(x=-155, y=1, z=50)
            ],
            'Town05_h50_2': [
                carla.Location(x=-50, y=0, z=50),
                carla.Location(x=-100, y=0, z=50),
                carla.Location(x=-20, y=0, z=50),
                carla.Location(x=-50, y=45, z=50)
            ],
            'Town05_h50_3': [
                carla.Location(x=-125, y=150, z=50),
                carla.Location(x=-125, y=90, z=50),
                carla.Location(x=-45, y=105, z=50),
                carla.Location(x=-190, y=125, z=50)
            ],
            ###################################################
            # Town03, 50m
            'Town03_h50_1': [
                carla.Location(x=43, y=-134, z=50),
                carla.Location(x=0, y=-85, z=50),
                carla.Location(x=-45, y=0, z=50),
                carla.Location(x=80, y=-85, z=50)
            ],
            'Town03_h50_2': [
                carla.Location(x=0, y=0, z=50),
                carla.Location(x=0, y=-50, z=50),
                carla.Location(x=0, y=50, z=50),
                carla.Location(x=50, y=0, z=50)
            ],
            'Town03_h50_3': [
                carla.Location(x=0, y=133, z=50),
                carla.Location(x=0, y=183, z=50),
                carla.Location(x=0, y=83, z=50),
                carla.Location(x=50, y=133, z=50)
            ],
            'Town03_h50_4': [
                carla.Location(x=-85, y=0, z=50),
                carla.Location(x=-40, y=0, z=50),
                carla.Location(x=-85, y=-50, z=50),
                carla.Location(x=-126, y=14, z=50)
            ],
            'Town03_h50_5': [
                carla.Location(x=171, y=60, z=50),
                carla.Location(x=150, y=130, z=50),
                carla.Location(x=235, y=-60, z=50),
                carla.Location(x=115, y=60, z=50)
            ],
            "Town03_h50_6": [
                carla.Location(x=-6, y=22, z=50),
                carla.Location(x=23, y=-4, z=50),
                carla.Location(x=0, y=-22, z=50),
                carla.Location(x=-36, y=-1, z=50)
            ],
            "Town03_h50_7": [
                carla.Location(x=-80, y=-30, z=50),
                carla.Location(x=23, y=-4, z=50),
                carla.Location(x=0, y=-22, z=50),
                carla.Location(x=-36, y=-1, z=50)
            ],

        }

        save_dir = os.path.join(base_dir, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Running on {args.town}")
        town_name = args.town.split('_')[0] if '_' in args.town else args.town
        main(town_name, 20, save_dir, test_points[args.town], args.random_seed)
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')