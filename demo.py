import os
import numpy as np
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics_rule_base import SimplePopulationDynamicsRuleBase
from garl_gym import scenarios
#from garl_gym.scenarios.simple_population_dynamics_fast import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

get_obs = scenarios.simple_population_dynamics_rule_base.get_obs
#args = {'predator_num': 5000, 'prey_num': 10000, 'num_actions': 4, 'height': 1000, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 2, 'max_speed': 1, 'max_acceleration': 1,
        #'width': 1000, 'batch_size': 32, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3}

args = {'agent_emb_dim': 5, 'obs_type': 'conv', 'predator_num': 1000, 'prey_num': 1000, 'num_actions': 4, 'height':500, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 5, 'max_speed': 1, 'max_acceleration': 1,
        'width': 500, 'batch_size': 512, 'vision_width': 21, 'vision_height': 21, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3, 'wall_prob': 0.05, 'wall_seed': 20, 'food_prob': 0, 'cpu_cores': 3, 'multiprocessing': True}
        #'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','']}
args = AttrDict(args)

env = SimplePopulationDynamicsRuleBase(args)
env.make_world(wall_prob=0.0005, wall_seed=20, food_prob=0)
#env.plot_map()

predators = []
preys = []
total_rewards = []
total_food = []

sum_rewards = 0

def take_actions(env):
    actions = {}
    for id, agent in env.agents.items():
        actions[id] = np.random.randint(4)
    return actions


for i in range(100000):
    print('Iteration: {:d} #Preys {:d} #Predators {:d} #food {:d}'.format(i, len(env.preys), len(env.predators), env.num_food))
    if i % 10 == 0:
        predators.append(len(env.predators))
        preys.append(len(env.preys))
        total_food.append(env.num_food)
        total_rewards.append(sum_rewards/10.)
        sum_rewards = 0
    #actions = take_actions(env)
    env.take_actions()
    view, rewards, killed = get_obs(env)
    env.killed = killed
    #obs, rewards = env.step(actions)
    #sum_rewards += np.sum(rewards)
    env.remove_dead_agents()
   # env.plot_map_cv2()
    env.dump_image('./tmp/{:d}.png'.format(i))
    if cv2.waitKey(100) & 0xFF == 27:
        break
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

 #   if len(env.predators) < 2:
 #       env.increase_predator(0.002)
 #   elif len(env.preys)<2:
 #       env.increase_prey(0.0001)

    #env.crossover_prey(crossover_rate=0.01)
    #env.crossover_predator(crossover_rate=0.01)
    env.crossover_prey(11, crossover_rate=0.012)
    env.crossover_predator(11, crossover_rate=0.006)

    if len(env.predators)<2 or len(env.preys) < 2 or len(env.predators) > 15000 or len(env.preys) > 15000:
        break
 #   env.increase_prey(0.006)
 #   env.increase_predator(0.003)
    #env.increase_food(prob=0.005)

sns.set_style("darkgrid")
plt.plot(list(range(0, i, 10)), predators)
plt.plot(list(range(0, i, 10)), preys)
#plt.plot(list(range(0, i, 10)), total_food)
plt.legend(['predators', 'preys'])
plt.show()
#
#plt.figure()
#plt.plot(total_rewards)
#plt.show()
plt.savefig('dynamics.png')

