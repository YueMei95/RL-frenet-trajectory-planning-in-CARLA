import os.path as osp
import os
import inspect
import sys
import matplotlib.pyplot as plt

# currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines.bench.monitor import load_results

agent_id = 25
n_smooth = 100
col = 'r'       # 'r', 'l', 'max index'

df = load_results('logs/agent_' + str(agent_id) + '/')

# Calculate timestep for each episode using episodes length
timestep = [df['l'][0]]
for i, epslen in enumerate(df['l'][1:]):
    timestep.append(epslen + timestep[i])
df['timestep'] = timestep       # add timestep column to df

print(df.head())
data_n_step = []
n_step = []
for i in range(df.shape[0]-n_smooth):
    data_n_step.append(df[col][i:i+n_smooth].mean())
    n_step.append(df['timestep'][i+n_smooth-1])

# Plot training & validation loss values
plt.plot(n_step, data_n_step)
plt.title('episodic ' + col)
plt.ylabel(col)
plt.xlabel('timestep')
# plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(True)
# plt.xticks(np.arange(1, 100, 5))
plt.show()
