import glob
import os
import sys
import time
import carla
from numpy import random

client = carla.Client('localhost', port = 2000)
client.set_timeout(5.0)

def print_spectator_location(world_snapshot):
    transform = spectator.get_transform()
    location = transform.location
    print(f"Spectator Location: x={location.x}, y={location.y}, z={location.z}")


# 选择你想要的地图
world = client.load_world_if_different('Town03')
if not world:
    world = client.get_world()



# 获取 Spectator
spectator = world.get_spectator()
world.on_tick(print_spectator_location)

import time
while True:
    time.sleep(1)