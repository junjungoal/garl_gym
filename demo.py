import os
import numpy as np
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
#import matplotlib.pyplot as plt
import cv2

args = {'predator_num': 10, 'prey_num': 20, 'num_actions': 5, 'height': 70, 'damage_per_step': 0.01, 'img_length': 5, 'hunt_radius': np.sqrt(8),
        'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','2500-5-5-3']}
args = AttrDict(args)

env = SimplePopulationDynamics(args)
env.make_world(wall_prob=0.02)
#env.plot_map()

for i in range(2000):
    if i % 10 == 0:
        print('time: {:d}'.format(i))
    actions = np.random.randint(5, size=10)
    rewards = env.step(actions)
    env.plot_map_cv2()
    if cv2.waitKey(100) & 0xFF == 27:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

print(env.predators,env.preys)
