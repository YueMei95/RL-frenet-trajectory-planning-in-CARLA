# initialize project
1. Clone the project
2. cd agents/reinforcement_learning
3. pip install -e .

# command help:
python run.py --env=CarlaGymEnv-v95 --num_timesteps=200e3

Runs last recorded agent_210 in specified environment 
python run.py --env=CarlaGymEnv-v553 --agent_id=210  --play_mode=1 --verbosity=2 --test_last --test

kubectl exec sim-carla-km2lm -c carla-client -- /bin/bash -c "cd carla-decison-making && python run_carla_stable.py --num_timesteps=1e6 --action_noise=0.0 --agent_id=1 |& tee /carla/models/1-output.txt"

kubectl cp data-transfer:/carla/CARLA-RL/b25c2e6/carla-decison-making/logs /home/asl/Desktop/CARLA_0.9.6/test_repo/logs

kubectl cp ucsc-adas/carla-sim-dn9nz:/carla/CARLA-RL/e106fb8/carla-decison-making/logs /Users/engintekin/Desktop/log -c carla-client

python config.py --weather ClearNoon -m Town04; 
# carla-decison-making
Long short term decision making for autonomous vehicles using depp reinforcement learning


export UE4_ROOT=~/UnrealEngine_4.22

DISPLAY= ./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20 -opengl -carla settings=CarlaSettings.ini

DISPLAY= ./CarlaUE4.sh /Game/Carla/Maps/Town02 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

DISPLAY= ./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

python3 client_controller.py -h   ===> for help

python3 client_controller.py -s 45 -cont pd -rep log_test

===================================================================================================

### Use pre-compiled carla versions - Example for CARLA 9.9.2
1. Download the pre-compiled CARLA simulator from [CARLA releases page](https://github.com/carla-simulator/carla/releases)
2. Now you can run this version using ./CarlaUE4.sh command
3. Create a virtual Python environemnt, e.g. using conda create -n carla99, and activate the environment, i.e. conda activate carla99
4. If easy_install is not installed already, run this: sudo apt-get install python-setuptools
5. Navigate to PythonAPI/carla/dist
6. Install carla as a python package into your virtual environment ([get help](https://carla.readthedocs.io/en/latest/build_system/)): easy_install --user --no-deps carla-X.X.X-py3.7-linux-x86_64.egg

Now you may import carla in your python script.

Player:
https://carla.readthedocs.io/en/latest/python_api_tutorial/#spawning-actors

import carla

### define carla world:
client = carla.Client(host, port)

word = client.get_world()

### blue print for each actor
blueprints = self.world.get_blueprint_library().filter('vehicle.*')[0]  # returns available blueprints for vehicles

blueprint = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0] # set the blueprint for the actor

blueprint.set_attribute('role_name', 'any string here is acceptable') # to distinguish actors

blueprint.set_attribute('color', '140,0,0')

### To define a transform:
transform = carla.Transform(carla.Location(x=402.242, y=-97.558, z=1.20001), carla.Rotation(pitch=0, yaw=-89.4014, roll=0))

### Create actor:
player = world.spawn_actor(blueprint, transform)   # generates error if infeasible; i.e. actor has collision with other actors
### or:
player = world.try_spawn_actor(blueprint, transform) # returns None if infeasible

### get actor transform:
tr = player.get_transform()

### change transform:
tr.location.y -= 3

tr.rotation.yaw = 0

player.set_transform(tr)

### send out controls to actors:
control = carla.VehicleControl()

control.throttle = 0.7

control.steer = 0

player.apply_control(control)


