#!/usr/bin/env python

import glob
import os
import sys
import time
import carla
import numpy as np
from numpy import random
from uav_utils import UAV

vehicles_list = []

# 设置CARLA客户端连接参数
host = '127.0.0.1'
port = 2000
synchronous_mode = True
tm_port = 8000

client = carla.Client(host, port)
client.set_timeout(10.0)

world = client.get_world()

# 更改地图
world = client.load_world_if_different('Town03')
if not world:
    world = client.get_world()

traffic_manager = client.get_trafficmanager(tm_port)

settings = world.get_settings()

default_substep = settings.max_substeps
default_substep_delta_time = settings.max_substep_delta_time

print("default_substep:", default_substep)
print("default_substep_delta_time:", default_substep_delta_time)

if synchronous_mode:
    traffic_manager.set_synchronous_mode(True)
    settings.synchronous_mode = True
    settings.max_substep_delta_time = 0.02
    settings.max_substeps = 10 # 1-16
    settings.fixed_delta_seconds = 0.2 # < 0.5
world.apply_settings(settings)

total_sec = 30
total_tick = int(total_sec / settings.fixed_delta_seconds) + 1

# 以上为基础设置
#####################################################################
def get_actor_blueprints(world, filter):
    return world.get_blueprint_library().filter(filter)



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
        uavs = []

        location1 = carla.Location(x=10, y=20, z=50)
        uavs.append(UAV(world, location1, uav_id=1, yaw_angle=0))
        uavs[0].set_world_origin([10, 20, 50])

        # delta_location = carla.Location(x=5, y=0, z=0)
        # uavs[0].enable_movement(True)
        # uavs[0].set_delta_location(delta_location)

        # location2 = carla.Location(x=50, y=50, z=50)
        # uavs.append(UAV(world, location2, uav_id=2, yaw_angle=90))

        # location3 = carla.Location(x=0, y=50, z=50)
        # uavs.append(UAV(world, location3, uav_id=3, yaw_angle=180))

        # location4 = carla.Location(x=0, y=0, z=50)
        # uavs.append(UAV(world, location4, uav_id=4, yaw_angle=270))

                

        # 参考co-perception学习一下传感器相关设置
        # 传感器位深、分辨率、视野fov、如何将深度图转换为点云
        # 让无人机进行移动
        # 让传感器有物理距离
        # 明确一下无人机的内外参
        # 采集信息的时候要包括无人机的内外参
        # 再学习一下数据的命名格式





#################################################################################
        # 开始运行        
        tick_interval = 1.0 / 30  # 渲染帧率

        tick_count = 0

        while True:
            start_time = time.time()

            if synchronous_mode:
                world.tick()
                tick_count += 1
                for uav in uavs:
                    uav.update()
            else:
                world.wait_for_tick()

            # 控制tick频率
            elapsed_time = time.time() - start_time
            if elapsed_time < tick_interval:
                time.sleep(tick_interval - elapsed_time)

            if tick_count > total_tick:
                break


    finally:# 结束运行
        all_actors = world.get_actors()

        # 筛选出载具（通过actor类型为carla.Vehicle）
        vehicles = all_actors.filter('vehicle.*')
        for vehicle in vehicles:
            print(f"Vehicle ID: {vehicle.id}")
            print(f"Vehicle Type: {vehicle.type_id}")
            print(f"Vehicle Location: {vehicle.get_location()}")
            print(f"Vehicle Velocity: {vehicle.get_velocity()}")
            print(f"Vehicle Acceleration: {vehicle.get_acceleration()}")
            print(f"Vehicle Bounding Box: {vehicle.bounding_box}")
            print("-" * 30)

        time.sleep(0.5)

        if synchronous_mode and world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.max_substep_delta_time = default_substep_delta_time
            settings.max_substeps = default_substep
            world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        for uav in uavs:
            uav.destroy()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

# 无人机的路径规划可以添加随机扰动（高斯噪声），以追求现实不稳定的飞行状况