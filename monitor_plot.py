import os.path as osp
import os
import inspect
import sys
import matplotlib.pyplot as plt

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines.bench.monitor import load_results

agent_id = 1
df = load_results('logs/agent_' + str(agent_id) + '/')

# Calculate timestep for each episode using episodes length
timestep = [df['l'][0]]
for i, epslen in enumerate(df['l'][1:]):
    timestep.append(epslen + timestep[i])
df['timestep'] = timestep       # add timestep column to df

print(df.head())

# Plot training & validation loss values
plt.plot(df['timestep'], df['r'])
plt.title('episodic reward')
plt.ylabel('rew')
plt.xlabel('timestep')
# plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
# plt.xticks(np.arange(1, 100, 5))
plt.show()
