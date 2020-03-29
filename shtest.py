from subprocess import Popen
import os.path as osp
import os
import inspect
import sys
import subprocess
import time

from pathlib import Path
#
# currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# parrentPath = Path(currentPath).parent
# print(currentPath)
# print(parrentPath)
# carlaPath = '.' + str(parrentPath) + '/CarlaUE4.sh'
# print(carlaPath)
# Process = Popen(carlaPath, shell=False)


import os

source = os.path.dirname(__file__)
parent = os.path.join(source, '../')
command = os.path.join(parent, 'CarlaUE4.sh ' + 'Town04 -quality-level=low -windowed -world-port=2000  '
                                                '-benchmark -fps=20 -opengl -carla-settings={}CarlaSettings.ini'.format(parent))
# DISPLAY= ./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20 -opengl -carla-settings=CarlaSettings.ini
# command = command.split()
# command[0] = 'DISPLAY= ' + command[0]
command = 'DISPLAY= ' + command
print(command)
process = Popen(command, shell=True, stdout=subprocess.PIPE)
time.sleep(5)
# print(process.pid)
# time.sleep(10)
# print('STDOUT:{}'.format(stdout))
# process.kill()

import psutil

# for proc in psutil.process_iter():
#     print(proc)

from subprocess import check_output

carla_process_ids = list(map(int, check_output(["pidof", 'CarlaUE4-Linux-Shipping']).decode("utf-8").split()))

print(carla_process_ids)

for pid in carla_process_ids:
    psutil.Process(pid).terminate()






