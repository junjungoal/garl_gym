import os, sys

import random
import multiprocessing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, imread, resize
import cv2
from copy import deepcopy
from garl_gym.base import BaseEnv

from garl_gym.core import DiscreteWorld, Agent


class SimplePopulationDynamics(BaseEnv):
    '''
    args:
        - height
        - width
        - batch_size
        - view_args
        - agent_number
        - num_actions ... not necessary(add flag?)
        - damage_per_step

    In the future, we define the attack ?
    If continuous, then size and speed?
    '''

    def __init__(self, args):
        self.args = args

        self.h = args.height
        self.w = args.width

        self.batch_size = args.batch_size
        if hasattr(args, 'view_args'):
            self.view_args = args.view_args
        else:
            self.view_args = None

        self.agent_num = args.predator_num
        self.predator_num = args.predator_num
        self.prey_num = args.prey_num
        self.action_num = args.num_actions


        # might need to be varied, depending on individuals
        self.vision_width = args.vision_width
        self.vision_height = args.vision_height

        self.ids = []
        self.map = np.zeros((self.h, self.w), dtype=np.int32)
        self.food_map = np.zeros((self.h, self.w), dtype=np.int32)
        self.property = {}

        self.killed = []

        # Health
        self.max_health = args.max_health
        self.min_health = args.min_health

        self.max_id = 0

        self.rewards = None

        self.max_view_size = None
        self.min_view_size = None
        self._init_property()

        self.predator_ids = []
        self.prey_ids = []

        self.max_hunt_square = args.max_hunt_square
        self.max_speed = args.max_speed
        self.max_crossover = args.max_crossover
        self.timestep = 0
        self.num_food = 0

    #@property
    #def predators(self):
    #    return self.agents[:self.predator_num]

    #@property
    #def preys(self):
    #    return self.agents[self.predator_num:]

    @property
    def agents(self):
        return self.predators + self.preys



    def make_world(self, wall_prob=0, wall_seed=10, food_prob=0.1, food_seed=10):
        self.gen_wall(wall_prob, wall_seed)
        self.gen_food(food_prob, food_seed)

        agents = [Agent() for _ in range(self.predator_num + self.prey_num)]

        for i, agent in enumerate(agents):
            agent.name = 'agent {:d}'.format(i+1)
            health = np.random.uniform(self.min_health, self.max_health)
            agent.health = health
            agent.original_health = health
            agent.birth_time = self.timestep
            if i < self.predator_num:
                agent.predator = True
                agent.id = i+1
                self.ids.append(i+1)
                self.predator_ids.append(agent.id)
                agent.speed = np.random.choice(self.max_speed) + 1
                agent.hunt_square = self.max_hunt_square - agent.speed+1
                agent.property = [self._gen_power(i+1), [0, 0, 1]]
            else:
                agent.predator = False
                agent.random = True # not trainable
                agent.id = -i-2
                self.prey_ids.append(agent.id)

            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    if agent.predator:
                        self.map[x][y] = i+1
                    else:
                        # we don't need to recognize each individual in prey in this environment
                        self.map[x][y] = -i-2
                    agent.pos = (x, y)
                    break

            self.predators = agents[:self.predator_num]
            self.preys = agents[self.predator_num:]

    def gen_food(self, prob=0.1, seed=10):
        for i in range(self.h):
            for j in range(self.w):
                food_prob = np.random.rand()
                if food_prob < prob and self.map[i][j] != -1 and self.food_map[i][j] == 0:
                    self.food_map[i][j] = -2
                    self.num_food += 1



    def gen_wall(self, prob=0, seed=10):
        if prob == 0:
            return
        np.random.seed(seed)

        for i in range(self.h):
            for j in range(self.w):
                if i == 0 or i == self.h-1 or j == 0 or j == self.w - 1:
                    self.map[i][j] = -1
                    continue
                wall_prob = np.random.rand()
                if wall_prob < prob:
                    self.map[i][j] = -1

    def _init_property(self):
        self.property[-3] = [1, [1, 0, 0]]
        self.property[-2] = [1, [0, 1, 0]]
        self.property[-1] = [1, [0, 0, 0]]
        self.property[0] = [1, [0.411, 0.411, 0.411]]

    def _gen_power(self, cnt):
        def max_view_size(view_size1, view_size2):
            view_size_area1 = (2*view_size1[0]+1) * (view_size1[1]+1)
            view_size_area2 = (2*view_size2[0]+1) * (view_size2[1]+1)
            return view_size1 if view_size_area1 > view_size_area2 else view_size2

        def min_view_size(view_size1, view_size2):
            view_size_area1 = (2*view_size1[0]+1) * (view_size1[1]+1)
            view_size_area2 = (2*view_size2[0]+1) * (view_size2[1]+1)
            return view_size1 if view_size_area1 < view_size_area2 else view_size2

        cur = 0
        if self.view_args is None:
            return [5, 5, 0]
        for k in self.view_args:
            k = [int(x) for x in k.split('-')]
            assert len(k) == 4
            num, power_list = k[0], k[1:]
            # Maintain the max_view_size
            if self.max_view_size is None:
                self.max_view_size = power_list
            else:
                self.max_view_size = max_view_size(self.max_view_size, power_list)

            if self.min_view_size is None:
                self.min_view_size = power_list
            else:
                self.min_view_size = min_view_size(self.min_view_size, power_list)

            cur += num

            if cnt <= cur:
                return power_list

    def crossover_predators(self, mate_scope=20, crossover_rate=0.01):
        for predator in self.predators:
            if np.random.rand() < crossover_rate and predator.age > 10 and predator.cross_over==False:
                x, y = predator.pos
                local_map = self.map[x-mate_scope-1:x+mate_scope, y-mate_scope-1:y+mate_scope]
                ind = np.where(local_map >0)
                max_fitness =  -np.inf
                mate_target = None
                for (x_local, y_local) in zip(ind[0], ind[1]):
                    id_ = local_map[x_local, y_local]
                    target_predator = self.predators[self.predator_ids.index(id_)]
                    if target_predator.cross_over==True:
                        continue
                    fitness = target_predator.max_reward
                    if max_fitness < fitness:
                        mate_target = target_predator
                        max_fitness = fitness

                if mate_target is not None:
                    target_predator.cross_over=True
                    new_predator = Agent()
                    available_ind = np.where(self.map==0)
                    while True:
                        new_id = np.random.randint(2, self.h*self.w)
                        if new_id not in self.predator_ids:
                            new_predator.id = new_id
                            new_predator.predator = True
                            ind_loc = np.random.randint(0, len(available_ind[0]))
                            new_x = max(x-mate_scope, 0)+available_ind[0][ind_loc]-1
                            new_y = max(y-mate_scope, 0)+available_ind[1][ind_loc]-1
                            ratio = np.random.uniform()
                            #health = predator.original_health*ratio + target_predator.original_health*(1-ratio)
                            health = 1
                            new_predator.health = health
                            new_predator.original_health = health
                            new_predator.property = [self._gen_power(new_predator.id), [0, 0, 1]]

                            # change
                            new_predator.speed = predator.speed
                            new_predator.hunt_square = predator.hunt_square

                            self.map[new_x, new_y] = new_id
                            new_predator.pos = (new_x, new_y)
                            self.predator_ids.append(new_id)
                            self.ids.append(new_id)
                            self.predators.append(new_predator)
                            break



    def crossover_preys(self, mate_scope=10, crossover_rate=0.01):
        for prey in self.preys:
            #if np.random.rand() < crossover_rate and prey.age > 10:
            if np.random.rand() < crossover_rate and prey.age > 10 and prey.cross_over == False:
                x, y = prey.pos
                local_map = self.map[x-mate_scope-1:x+mate_scope, y-mate_scope-1:y+mate_scope]
                ind = np.where(local_map < -2)
                min_dist = np.inf
                mate_target = None
                available_ind = np.where(local_map == 0)
                if len(available_ind[0]) == 0:
                    continue
                for (x_local, y_local) in zip(ind[0], ind[1]):
                    id_ = local_map[x_local, y_local]
                    target_prey = self.preys[self.prey_ids.index(id_)]
                    if target_prey.cross_over==True or target_prey.age <= 10:
                        continue
                    x_target, y_target = target_prey.pos
                    dist = (x-x_target)**2+(y-y_target)**2
                    if dist < min_dist:
                        mate_target = target_prey
                        min_dist = dist

                if mate_target is not None:
                    mate_target.cross_over=True
                   # for _ in range(np.random.choice(self.max_crossover)+1):
                    new_prey = Agent()
                    while True:
                        new_id = -np.random.randint(3, self.h*self.w)
                        if new_id not in self.prey_ids:
                            new_prey.id = new_id
                            new_prey.predator = False
                            new_prey.random=True
                            ind_loc = np.random.randint(0, len(available_ind[0]))
                            new_x = max(x-mate_scope, 0)+available_ind[0][ind_loc] - 1
                            new_y = max(y-mate_scope, 0)+available_ind[1][ind_loc] - 1
                            ratio = np.random.uniform()
                            #health = prey.original_health*ratio + mate_target.original_health*(1-ratio)
                            health = 1
                            new_prey.health = health
                            new_prey.original_health = health
                            self.map[new_x, new_y] = new_id
                            new_prey.pos = (new_x, new_y)
                            self.prey_ids.append(new_id)
                            self.preys.append(new_prey)
                            break



    def take_actions(self, actions):
        self.actions = actions
        for i, agent in enumerate(self.agents):
            if agent.predator:
                self._predator_action(agent, actions[i])
            else:
                self._prey_action(agent)
            self.decrease_health(agent)

    def _prey_action(self, agent):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        x, y = agent.pos
        direction = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        np.random.shuffle(direction)
        for pos_x, pos_y in direction:
            if (pos_x, pos_y) == (0, 0):
                break
            new_x = x + pos_x
            new_y = y + pos_y

            if in_board(new_x, new_y) and self.map[new_x][new_y] == 0:
                agent.pos = (new_x, new_y)
                self.map[new_x][new_y] = agent.id
                self.map[x][y] = 0
                if self.food_map[new_x, new_y] == -2:
                    self.food_map[new_x, new_y] = 0
                    agent.health += 0.2
                    self.num_food -= 1
                break



    def _predator_action(self, agent, action):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w) and self.map[x][y] == 0
        x, y = agent.pos
        if action == 0:
            new_x = x - agent.speed
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 1:
            new_x = x + agent.speed
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 2:
            new_x = x
            new_y = y - agent.speed
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 3:
            new_x = x
            new_y = y + agent.speed
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        else:
            print('Wrong action id')

        new_x, new_y = agent.pos
        self.map[x][y] = 0
        self.map[new_x][new_y] = agent.id

    def increase_food(self, prob):
        num = max(1, int(self.num_food * prob))
        for _ in range(num):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] != -1 and self.food_map[x][y] == 0:
                    self.food_map[x][y] = -2
                    self.num_food += 1
                    break


        ## Exclude Grouping

    def decrease_health(self, agent):
        #for i in range(self.predator_num):
        #for i in range(len(self.agents)):
            #self.agents[i].health -= self.args.damage_per_step
        agent.health -= self.args.damage_per_step

    def increase_health(self, agent):
        agent.health += 0.02


    def dump_image(self, img_name):
        new_w, new_h = self.w * 5, self.h * 5
        img = np.zeros((new_w, new_h, 3), dtype=np.uint8)
        length = self.args.img_length
        for i in range(self.w):
            for j in range(self.h):
                id = self.map[i][j]
                if id <= 0:
                    img[i * length:i*(length+1)][j * length:j*(length+1)] = 255 * np.array(self.property[id][1])
        for predator in self.predators:
            x, y = predator.pos
            img[x * length:x*(length+1)][y* length:y*(length+1)] = 255 * np.array(predator.property[1])
        output_img = Image.fromarray(img, 'RGB')
        output_img.save(img_name)

    def convert_img(self):
        img = np.zeros((self.h, self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                id = self.map[i][j]
                if self.food_map[i][j] == -2:
                    img[i, j, :] = 255*np.array(self.property[-2][1])
                elif id <= 0 and id > -2:
                    img[i, j, :] = 255*np.array(self.property[id][1])
                else:
                    # prey
                    img[i, j, :] = 255*np.array(self.property[-3][1])

        for predator in self.predators:
            x, y = predator.pos
            img[x, y, :] = 255*np.array(predator.property[1])
        return img


    def get_predator_reward(self, agent):
        preys = self.agents[self.predator_num:]
        reward = 0
        x, y = agent.pos
        local_map = self.map[x-agent.hunt_square-1:x+agent.hunt_square, y-agent.hunt_square-1:y+agent.hunt_square]
        id_prey_loc = np.where(local_map < -2)

        if len(id_prey_loc[0]) > 0:
            min_dist = np.inf
            target_prey = None
            for (local_x_prey, local_y_prey) in zip(id_prey_loc[0], id_prey_loc[1]):
                id_ = local_map[local_x_prey, local_y_prey]
                prey = self.preys[self.prey_ids.index(id_)]
                x_prey, y_prey = prey.pos
                dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
                if dist < min_dist:
                    min_dist = dist
                    target_prey = prey

            reward += 1
            target_prey.dead = True
            self.preys.remove(target_prey)
            self.prey_ids.remove(target_prey.id)
            self.prey_num -= 1
            x, y = target_prey.pos
            self.map[x][y] = 0
            agent.max_reward += 1
            self.increase_health(agent)
        return reward

    def remove_dead_agents(self):
        for agent in self.agents:
            if agent.health <= 0:
            #if agent.health <= 0:
                if agent.predator:
                    self.predators.remove(agent)
                    self.ids.remove(agent.id)
                    self.predator_ids.remove(agent.id)
                    self.predator_num -= 1
                else:
                    self.preys.remove(agent)
                    self.prey_num -= 1
                    self.prey_ids.remove(agent.id)
                x, y = agent.pos
                self.map[x][y] = 0
            else:
                agent.age += 1
                agent.crossover=False

    def reset_env(self):
        self.make_world()

    def _get_obs(self, agent):
        x, y = agent.pos
        obs = self.map[max((x-self.vision_width//2)-1, 0):min((x+self.vision_width//2), self.map.shape[0]), max((y-self.vision_height//2)-1, 0):min((y+self.vision_height//2), self.map.shape[1])]
        left_ex = abs(min((x-self.vision_width//2)-1, 0))
        right_ex = max(x+self.vision_width//2 - self.map.shape[0], 0)
        top_ex = abs(min((y-self.vision_height//2)-1, 0))
        bottom_ex = max(y+self.vision_height//2-self.map.shape[1], 0)
        obs = np.pad(obs, ((left_ex, right_ex), (top_ex, bottom_ex)), mode='constant', constant_values=-1).astype(np.float)
        return obs

    def step(self, actions):
        self.take_actions(actions)
        rewards = []
        obs = []
        self.killed = []
        for predator in self.predators:
            rewards.append(self.get_predator_reward(predator))

        for agent in self.agents:
            obs.append(self._get_obs(agent))

        self.remove_dead_agents()

        return obs, rewards
