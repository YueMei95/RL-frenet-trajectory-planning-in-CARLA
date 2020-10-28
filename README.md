# RL-frenet-trajectory-planning-in-CARLA
This repository is a framework that creates a OpenAI Gym environment for self-driving car simulator CARLA in order to utilize cutting edge deep reinforcement algorithms and frenet trajectory planning.
=======

# Installation
1. Clone the project
2. pip3 install -r requirements.txt (requires Python 3.7 or newer)
3. cd agents/reinforcement_learning
4. pip install -e . # installs RL algorithms as python packages 

# Use pre-compiled carla versions - (CARLA 9.9.2 Recommended)
1. Download the pre-compiled CARLA simulator from [CARLA releases page](https://github.com/carla-simulator/carla/releases)
2. Now you can run this version using ./CarlaUE4.sh command
3. Create a virtual Python environemnt, e.g. using conda create -n carla99, and activate the environment, i.e. conda activate carla99
4. If easy_install is not installed already, run this: sudo apt-get install python-setuptools
5. Navigate to PythonAPI/carla/dist
6. Install carla as a python package into your virtual environment ([get help](https://carla.readthedocs.io/en/latest/build_system/)): easy_install --user --no-deps carla-X.X.X-py3.7-linux-x86_64.egg

Now you may import carla in your python script.

# Some Features

- Simulation works as server-client. CARLA launches as server and uses 2000:2002 ports as default. Client can connect to server from port 2000, default, and interract with environment.
- Reinforcement Learning/Gym Environment parameters are configured at /tools/cfgs/config.yaml
- DDPG/TRPO/A2C/PPO2 are configured to save models during training with intervals and also best models with max_moving_average (window_size=100 default)

# Example Training:

- ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low [CARLA documentation](https://carla.readthedocs.io/en/latest/)
- python3 run.py --cfg_file=tools/cfgs/config.yaml --agent_id=1 --env=CarlaGymEnv-v5522 
 
# Example Test:

Initilize the best recorded agent and associated config file given the agent_id. Test runs as --play_mode=1 (2D) as default. 

- ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low
- python3 run.py --agent_id=1 --env=CarlaGymEnv-v5522 --test 

# Pre-trained agents DDPG(ID:1), TRPO(ID:2), A2C(ID:3), PPO2(ID:4)

Besides the config.yaml file you can also use following parameters:

--num_timesteps; number of the time steps to train agent, default=1e7 
--play_mode: Display mode: 0:off, 1:2D, 2:3D, default=0
--verbosity: 0:Off, 1:Action,Reward, 2: Actors + 1, 3: Observation Tensor + 2, default=0
--test: default=False
--test_model: if want to run a specific model type:str without file extension example (best_120238)
--test_last: if True will run the latest recorded model not the best

- Carla requires a powerful GPU to produce high fps. In order to increase performance you can run following as an alternative:

DISPLAY= ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low

# Important Directories
- RL Policy Networks : agents/reinforcement_learning/stable_baselines/common/policies.py
- Env and RL Config File: tools/cfgs/config.yaml
- Gym Environment: carla_gym/envs/ # Gym environment interface for CARLA, To manipulate observation, action, reward etc.
- Modules: tools/modules.py # Pretty much wraps everything
