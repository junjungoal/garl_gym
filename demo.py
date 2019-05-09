import os
import numpy as np
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics_ga_near import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

args = {'predator_num': 400, 'prey_num': 400, 'num_actions': 4, 'height': 120, 'damage_per_step': 0.02, 'img_length': 7, 'max_hunt_square': 3, 'max_speed': 3, 'max_acceleration': 3,
        'width': 120, 'batch_size': 1, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3}
        #'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','']}
args = AttrDict(args)

env = SimplePopulationDynamics(args)
env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0.1)
#env.plot_map()

predators = []
preys = []
total_rewards = []
total_food = []

sum_rewards = 0
for i in range(2000):
    print('Iteration: {:d} #Preys {:d} #Predators {:d} #food {:d}'.format(i, len(env.preys), len(env.predators), env.num_food))
    if i % 10 == 0:
        predators.append(len(env.predators))
        preys.append(len(env.preys))
        total_food.append(env.num_food)
        total_rewards.append(sum_rewards/10.)
        sum_rewards = 0
    actions = np.random.randint(4, size=len(env.predators))
    obs, rewards = env.step(actions)
    sum_rewards += np.sum(rewards)
    #env.plot_map_cv2()
    if cv2.waitKey(100) & 0xFF == 27:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    #if len(env.predators) < 2:
    env.increase_predator(0.002)
   # elif len(env.preys)<2:
    env.increase_prey(0.0001)

    env.crossover_preys(crossover_rate=0.6)
    env.crossover_predators(crossover_rate=0.01)
    #env.increase_prey(0.03)
    #env.increase_predator(0.006)
    env.increase_food(prob=0.005)

sns.set_style("darkgrid")
plt.plot(list(range(0, i, 10)), predators)
plt.plot(list(range(0, i, 10)), preys)
#plt.plot(list(range(0, i, 10)), total_food)
plt.legend(['predators', 'preys'])
plt.show()

plt.figure()
plt.plot(total_rewards)
plt.show()

print(env.predators,env.preys)
