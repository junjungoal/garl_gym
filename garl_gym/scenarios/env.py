import os, sys

import random
import multiprocessing
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, imread, resize
import cv2
from copy import deepcopy

from garl_env.base import BaseScenario


class SimplePopulationDynamics(BaseScenario):
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
        self.h = args.height # grid size height
        self.w = args.width # grid size width
        self.batch_size = args.batch_size
        self.view_args = args.view_args

        self.agent_num = args.agent_number
        self.tiger_num = 0
        self.pig_num = 0
        self.action_num = args.num_actions

        self.map = np.zeros((self.h, self.w), dtype=np.int32)
        self.id_pos = {}
        self.tiger_pos = set()
        self.pig_pos = set()
        self.property = {}

        # Health
        self.health = {}
        self.max_id = 0

        self.rewards = None
        self.reward_radius_pig = args.reward_radius_pig
        self.reward_threshold_pig = args.reward_threshold_pig

        self.max_view_size = None
        self.min_view_size = None

        self._init_property()

    def _init_property(self):
        self.property[-3] = [1, [0, 1, 0]]
        self.property[-2] = [1, [1, 0, 0]]
        self.property[-1] = [1, [0, 0, 0]]
        self.property[0] = [1, [0.411, 0.411, 0.411]]


    def gen_agent(self, agent_num=None):
        if agent_num == None:
            agent_num = self.args.agent_number

        for i in range(agent_num):
            while True:
                # initialzie the agent's posision randomly
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.map[x][y] = i+1
                    self.id_pos[i+1] = (x, y)
                    self.property[i + 1] = [self._gen_power(i+1), [0, 0, 1]] #gen_power?
                    self.health[i+1] = 1.0
                    # property, health, birth_year?
                    break
        # Assert

        self.agent_num = self.args.agent_number
        self.max_id = self.args.agent_number

    def gen_pig(self, pig_nums=None):
        if pig_nums == None:
            pig_nums = self.args.pig_max_number
        for i in range(pig_nums):
            while True:
                x = np.random.randint(0, self.h)
                y = np.random.randint(0, self.w)
                if self.map[x][y] == 0:
                    self.map[x][y] = -2
                    self.pig_pos.add((x, y))
                    break
        self.pig_num = self.pig_num + pig_nums

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

    def update_pig_pos(self):
        # movement is random.... why dont they try to escape with strategy?
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w)

        # Move Pigs
        for i, item in enumerate(self.pig_pos):
            x, y = item
            direction = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
            np.random.shuffle(direction)
            for pos_x, pos_y in direction:
                if (pos_x, pos_y) == (0, 0):
                    break
                new_x = x + pos_x
                new_y = y + pos_y

                if in_board(new_x, new_y) and self.map[new_x][new_y] == 0:
                    self.pig_pos.remove((x, y))
                    self.pig_pos.add((new_x, new_y))
                    self.map[new_x][new_y] = -2
                    self.map[x][y] = 0
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

    def _agent_act(self, x, y, face, action, id):
        '''
        Face: face direction
        '''
        def move_forward(x, y, face):
            if face == 0:
                return x - 1, y
            elif face == 1:
                return x, y + 1
            elif face == 2:
                return x + 1, y
            elif face == 3:
                return x, y - 1

        def move_backward(x, y, face):
            if face == 0:
                return x + 1, y
            elif face == 1:
                return x, y - 1
            elif face == 2:
                return x - 1, y
            elif face == 3:
                return x, y + 1

        def move_left(x, y, face):
            if face == 0:
                return x, y - 1
            elif face == 1:
                return x - 1, y
            elif face == 2:
                return x, y + 1
            elif face == 3:
                return x + 1, y

        def move_right(x, y, face):
            if face == 0:
                return x, y + 1
            elif face == 1:
                return x + 1, y
            elif face == 2:
                return x, y - 1
            elif face == 3:
                return x - 1, y
        def in_board(x, y):
            return self.map[x][y] == 0

        def max_view_size(view_size1, view_size2):
            view_size_area1 = (2 * view_size1[0] + 1) * (view_size1[1] + 1)
            view_size_area2 = (2 * view_size2[0] + 1) * (view_size2[1] + 1)

            return view_size1 if view_size_area1 > view_size_area2 else view_size2


        if action == 0:
            pass
        elif action == 1:
            new_x, new_y = move_forward(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 2:
            new_x, new_y = move_backward(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)

        elif action == 3:
            new_x, new_y = move_right(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)

        elif action == 4:
            new_x, new_y = move_left(x, y, face)
            if in_board(new_x, new_y):
                self.map[x][y] = 0
                self.map[new_x][new_y] = id
                self.id_pos[id] = (new_x, new_y)
        elif action == 5:
            self.property[id][0][2] = (face+4-1) % 4  # turn left
        elif action == 6:
            self.property[id][0][2] = (face+1) % 4 # turn right
        else:
            print(action)
            print("Wrong Action ID!!!!")

        ## Exclude Grouping

    def decrease_health(self):
        for id, _ in self.id_pos.items():
            self.health[id] -= self.args.damage_per_step



    def take_actions(self, actions):
        '''
        Call action for each agent
        '''
        self.actions = actions
        for id, action in actions:
            x, y = self.id_pos[id]
            face = self.property[id][0][2]
            self._agent_act(x, y, face, action, id)

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




    def make_video(self, images, outvid=None, fps=5, size=None, is_color=True, format='XVID'):
        raise NotImplementedError


    def dump_image(self, img_name):
        raise NotImplementedError

    def convert_img(self):
        img = np.zeros((self.h, self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                id = self.map[i][j]
                img[i, j, :] = 255*np.array(self.property[id][1])
        return img


    def plot_map(self):
        plt.figure(figsize=(10, 10))
        img = self.convert_img()
        plt.imshow(img, interpolation="nearest")
        #plt.imshow(self._layout > -1, interpolation="nearest")
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        h, w = self.h, self.w
        for y in range(h-1):
            plt.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
        for x in range(w-1):
            plt.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)

    def step(self, actions):
        self.take_actions(actions)
        self.decrease_health()
        self.update_pig_pos()

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        reward_ids_pig = pool.map(self.get_reward_pig, self.pig_pos)





