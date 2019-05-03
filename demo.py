import os
import numpy as np
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

args = {'predator_num': 20, 'prey_num': 40, 'num_actions': 5, 'height': 70, 'damage_per_step': 0.01, 'img_length': 5, 'hunt_radius': np.sqrt(8),
        'width': 70, 'batch_size': 1, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5}
        #'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','']}
args = AttrDict(args)

env = SimplePopulationDynamics(args)
env.make_world(wall_prob=0.02)
#env.plot_map()

predators = []
preys = []
total_rewards = []

sum_rewards = 0
for i in range(2000):
    if i % 10 == 0:
        print('time: {:d}'.format(i))
        predators.append(len(env.predators))
        preys.append(len(env.preys))
        total_rewards.append(sum_rewards/10.)
        sum_rewards = 0
    actions = np.random.randint(5, size=args.predator_num)
    obs, rewards = env.step(actions)
    sum_rewards += np.sum(rewards)
    env.plot_map_cv2()
    if cv2.waitKey(100) & 0xFF == 27:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
print(preys)
print(predators)

sns.set_style("darkgrid")
plt.plot(list(range(0, i, 10)), predators)
plt.plot(list(range(0, i, 10)), preys)
plt.legend(['predators', 'preys'])
plt.show()

plt.figure()
plt.plot(total_rewards)
plt.show()

print(env.predators,env.preys)
