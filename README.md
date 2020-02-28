# run:
python config.py --weather ClearNoon -m Town04

python run_carla_stable.py --num_timesteps=1e6

kubectl exec sim-carla-km2lm -c carla-client -- /bin/bash -c "cd carla-decison-making && python run_carla_stable.py --num_timesteps=1e6 --action_noise=0.0 --agent_id=1 |& tee /carla/models/1-output.txt"

# carla-decison-making
Long short term decision making for autonomous vehicles using depp reinforcement learning


export UE4_ROOT=~/UnrealEngine_4.22

DISPLAY= ./CarlaUE4.sh /Game/Carla/Maps/Town02 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

DISPLAY= ./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

./CarlaUE4.sh Town04 -quality-level=low -windowed -world-port=2000  -benchmark -fps=20

python3 client_controller.py -h   ===> for help

python3 client_controller.py -s 45 -cont pd -rep log_test

===================================================================================================

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


