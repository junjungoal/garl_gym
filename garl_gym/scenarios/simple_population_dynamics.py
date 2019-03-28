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
        self.view_args = args.view_args

        self.agent_num = args.predator_num
        self.predator_num = args.predator_num
        self.prey_num = args.prey_num
        self.action_num = args.num_actions

        self.agents = []
        self.ids = []
        self.map = np.zeros((self.h, self.w), dtype=np.int32)
        self.property = {}

        self.killed = []

        # Health
        self.max_id = 0

        self.rewards = None
        self.hunt_radius = args.hunt_radius

        self.max_view_size = None
        self.min_view_size = None
        self._init_property()

    @property
    def predators(self):
        return self.agents[:self.predator_num]

    @property
    def preys(self):
        return self.agents[self.predator_num:]


    def make_world(self, wall_prob=0, wall_seed=10):
        self.gen_wall(wall_prob, wall_seed)

        self.agents = [Agent() for _ in range(self.predator_num + self.prey_num)]

        for i, agent in enumerate(self.agents):
            agent.name = 'agent {:d}'.format(i+1)
            if i < self.predator_num:
                agent.predator = True
                agent.health = 1.0
                agent.id = i+1
                self.ids.append(i+1)
                agent.property = [self._gen_power(i+1), [0, 0, 1]]
            else:
                agent.predator = False
                agent.random = True # not trainable

            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    if agent.predator:
                        self.map[x][y] = i+1
                    else:
                        # we don't need to recognize each individual in prey in this environment
                        self.map[x][y] = -2
                    agent.pos = (x, y)
                    break

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
        self.property[-3] = [1, [0, 1, 0]]
        self.property[-2] = [1, [1, 0, 0]]
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

    def take_actions(self, actions):
        self.actions = actions
        for i, agent in enumerate(self.agents):
            if agent.predator:
                self._predator_action(agent, actions[i])
            else:
                self._prey_action(agent)


    def _prey_action(self, agent):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        x, y = agent.pos
        direction = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
        np.random.shuffle(direction)
        for pos_x, pos_y in direction:
            if (pos_x, pos_y) == (0, 0):
                break
            new_x = x + pos_x
            new_y = y + pos_y

            if in_board(new_x, new_y) and self.map[new_x][new_y] == 0:
                agent.pos = (new_x, new_y)
                self.map[new_x][new_y] = -2
                self.map[x][y] = 0
                break



    def _predator_action(self, agent, action):
        def in_board(x, y):
            return self.map[x][y] == 0
        x, y = agent.pos
        if action == 0:
            pass
        elif action == 1:
            new_x = x - 1
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 2:
            new_x = x + 1
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 3:
            new_x = x
            new_y = y - 1
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        elif action == 4:
            new_x = x
            new_y = y + 1
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
        else:
            print('Wrong action id')

        new_x, new_y = agent.pos
        self.map[x][y] = 0
        self.map[new_x][new_y] = agent.id


        ## Exclude Grouping

    def decrease_health(self):
        for i in range(self.predator_num):
            self.agents[i].health -= self.args.damage_per_step

    def increase_health(self, agent):
        agent.health += 1

    def get_reward_pig(self):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        x, y = self.pig_pos
        groups_num = {}
        for i in range(-self.reward_radius_pig, self.reward_radius_pig):
            for j in range(-self.reward_radius_pig, self.reward_radius_pig):
                new_x, new_y = x+i, y+j
                if in_board(new_x, new_y):
                    id = self.map[new_x][new_y]
                    ## No group
         #           if id > 0:


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
                if id <= 0:
                    img[i, j, :] = 255*np.array(self.property[id][1])
        for predator in self.predators:
            x, y = predator.pos
            img[x, y, :] = 255*np.array(predator.property[1])
        return img


    def get_predator_reward(self, agent):
        preys = self.agents[self.predator_num:]
        reward = 0
        for prey in preys:
            if np.sqrt(np.sum(np.square(np.subtract(agent.pos, prey.pos)))) <= self.hunt_radius and not prey.dead:
                reward += 1
                prey.dead = True
                self.agents.remove(prey)
                self.prey_num -= 1
                x, y = prey.pos
                self.map[x][y] = 0
                self.increase_health(agent)
        return reward

    def remove_dead_predators(self):
        for predator in self.predators:
            if predator.health <= 0:
                self.agents.remove(predator)
                self.ids.remove(predator.id)
                x, y = predator.pos
                self.predator_num -= 1
                self.map[x][y] = 0

    def reset_env(self):
        self.make_world()

    def _get_obs(self):
        raise NotImplementedError

    def step(self, actions):
        self.take_actions(actions)
        self.decrease_health()
        rewards = []
        self.killed = []
        for predator in self.predators:
            rewards.append(self.get_predator_reward(predator))

        self.remove_dead_predators()


        return rewards

