import glob
import os
import sys
import time
import carla
from numpy import random

client = carla.Client('localhost', port = 2000)
client.set_timeout(5.0)

# 选择你想要的地图
world = client.load_world_if_different('Town03')
if not world:
    world = client.get_world()

