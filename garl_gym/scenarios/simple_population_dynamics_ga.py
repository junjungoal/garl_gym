import os, sys

import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from cv2 import VideoWriter, imread, resize
import cv2
from copy import deepcopy
from garl_gym.base import BaseEnv
import multiprocessing as mp
import gc
from garl_gym.core import DiscreteWorld, Agent


class SimplePopulationDynamicsGA(BaseEnv):
    '''
    args:
        - height
        - width
        - batch_size
        - view_args
        - agent_numbee
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

        self.map = np.zeros((self.h, self.w), dtype=np.int32)
        self.food_map = np.zeros((self.h, self.w), dtype=np.int32)
        self.property = {}

        self.killed = []

        # Health
        self.max_health = args.max_health
        self.min_health = args.min_health

        self.max_id = 1

        self.rewards = None

        self.max_view_size = None
        self.min_view_size = None
        self._init_property()

        self.get_closer_reward = args.get_closer_reward


        self.max_hunt_square = args.max_hunt_square
        self.max_speed = args.max_speed
        self.max_ = args.max_crossover
        self.timestep = 0
        self.num_food = 0

        self.obs_type = args.obs_type

        self.agent_embeddings = {}
        self.agent_emb_dim = args.agent_emb_dim

        self.cpu_cores = args.cpu_cores



    #@property
    #def predators(self):
    #    return self.agents[:self.predator_num]

    #@property
    #def preys(self):
    #    return self.agents[self.predator_num:]

    @property
    def agents(self):
        return {**self.predators, **self.preys}



    def make_world(self, wall_prob=0, wall_seed=10, food_prob=0.1, food_seed=10):
        self.gen_wall(wall_prob, wall_seed)
        self.gen_food(food_prob, food_seed)

        predators = {}
        preys = {}

        agents = [Agent() for _ in range(self.predator_num + self.prey_num)]

        empty_cells_ind = np.where(self.map == 0)
        perm = np.random.permutation(range(len(empty_cells_ind[0])))

        for i, agent in enumerate(agents):
            agent.name = 'agent {:d}'.format(i+1)
            health = np.random.uniform(self.min_health, self.max_health)
            agent.health = health
            agent.original_health = health
            agent.birth_time = self.timestep
            if i < self.predator_num:
                agent.predator = True
                agent.id = self.max_id
                agent.speed = 1
                agent.hunt_square = self.max_hunt_square
                agent.property = [self._gen_power(i+1), [0, 0, 1]]
            else:
                agent.predator = False
                agent.id = i+1
                agent.property = [self._gen_power(i+1), [1, 0, 0]]
            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[agent.id] = new_embedding

            x = empty_cells_ind[0][perm[i]]
            y = empty_cells_ind[1][perm[i]]
            self.map[x][y] = self.max_id
            agent.pos = (x, y)
            self.max_id += 1

            if agent.predator:
                predators[agent.id] = agent
            else:
                preys[agent.id] = agent

            self.predators = predators
            self.preys = preys
            #lproxy = mp.Manager().list()
            #lproxy.append({})
            #lproxy.append({})
            #self.l_predators = lproxy[0]
            #self.l_preys = lproxy[1]
            #self.l_predators = self.predators
            #self.l_preys = self.preys


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
                #if i == 0 or i == self.h-1 or j == 0 or j == self.w - 1:
                #    self.map[i][j] = -1
                #    continue
                wall_prob = np.random.rand()
                buffer = []
                connected_wall = []
                if wall_prob < prob:
                    #self.map[i][j] = -1
                    buffer.append((i, j))
                    connected_wall.append((i, j))

                    while len(buffer) != 0:
                        coord = buffer.pop()
                        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                if np.random.rand() < 0.15 and coord[0]+x>=0 and coord[0]+x<=self.h-1 and coord[1]+y>=0 and coord[1]+y<=self.w-1:
                                    buffer.append((coord[0]+x, coord[1]+y))
                                    connected_wall.append((coord[0]+x, coord[1]+y))
                                    self.map[coord[0]+x][coord[1]+y] = -1
                    if len(connected_wall) > 1:
                        for (x, y) in connected_wall:
                            self.map[x][y] = -1

    def _init_property(self):
        self.property[-3] = [1, [1, 0, 0]]
        self.property[-2] = [1, [0, 1, 0]]
        self.property[-1] = [1, [0.411, 0.411, 0.411]]
        self.property[0] = [1, [0, 0, 0]]

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

    def increase_food(self, prob):
        num = max(1, int(self.num_food * prob))
        ind = np.where(self.food_map==0)
        num = min(num, len(ind[0]))
        perm = np.random.permutation(np.arange(len(ind[0])))
        for i in range(num):
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            if self.map[x][y] != -1 and self.food_map[x][y] == 0:
                self.food_map[x][y] = -2
                self.num_food += 1

    def crossover_predator(self, crossover_scope=3, crossover_rate=0.001):
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0
        for predator in list(self.predators.values()):
            x, y = predator.pos
            local_map = self.map[(x-crossover_scope//2):(x-crossover_scope//2+crossover_scope), (y-crossover_scope//2):(y-crossover_scope//2+crossover_scope)]
            agent_indice = np.where(local_map > 0)
            if len(agent_indice[0]) == 0 or predator.crossover:
                continue

            for (local_x, local_y) in zip(agent_indice[0], agent_indice[1]):
                candidate_id = local_map[local_x, local_y]
                candidate_agent = self.agents[candidate_id]
                if candidate_agent.predator and not candidate_agent.crossover and predator.id != candidate_agent.id:
                    candidate_agent.get_closer = True
                    if np.random.rand() < crossover_rate:
                        for i in range(np.random.randint(self.args.max_predator_offsprings)):
                            candidate_agent.crossover = True
                            predator.crossover = True
                            child = Agent()
                            child.id = self.max_id
                            self.max_id += 1
                            new_embedding = np.random.normal(size=[self.agent_emb_dim])
                            self.agent_embeddings[child.id] = new_embedding
                            child.spped = None
                            child.predator = True
                            child.health = 1
                            child.hunt_square = self.max_hunt_square
                            child.property = [self._gen_power(child.id), [0, 0, 1]]
                            x = ind[0][perm[index]]
                            y = ind[1][perm[index]]
                            index += 1
                            self.map[x][y] = child.id
                            child.pos = (x, y)
                            self.predators[child.id] = child
                            self.predator_num += 1

    def crossover_prey(self, crossover_scope=3, crossover_rate=0.001):
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0
        for prey in list(self.preys.values()):
            x, y = prey.pos
            left = np.maximum(x-crossover_scope//2, 0)
            right = np.minimum(x-crossover_scope//2+crossover_scope, self.w)
            top = np.maximum(y-crossover_scope//2, 0)
            bottom = np.minimum(y-crossover_scope//2+crossover_scope, self.h)
            local_map = self.map[left:right, top:bottom]
            agent_indice = np.where(local_map > 0)
            if len(agent_indice[0]) == 0 and not prey.crossover:
                continue

            for (local_x, local_y) in zip(agent_indice[0], agent_indice[1]):
                candidate_id = local_map[local_x, local_y]
                candidate_agent = self.agents[candidate_id]
                candidate_agent.get_closer = True
                if not candidate_agent.predator and not candidate_agent.crossover and candidate_agent.id != prey.id:
                    if np.random.rand() < crossover_rate:
                        for i in range(np.random.randint(self.args.max_prey_offsprings)):
                            candidate_agent.crossover = True
                            prey.crossover = True
                            child = Agent()
                            child.id = self.max_id
                            self.max_id += 1
                            child.speed = None
                            child.predator = False
                            child.health = 1
                            new_embedding = np.random.normal(size=[self.agent_emb_dim])
                            self.agent_embeddings[child.id] = new_embedding
                            child.hunt_square = self.max_hunt_square
                            child.property = [self._gen_power(child.id), [1, 0, 0]]
                            x = ind[0][perm[index]]
                            y = ind[1][perm[index]]
                            index += 1
                            self.map[x][y] = child.id
                            child.pos = (x, y)
                            self.preys[child.id] = child
                            self.prey_num += 1


    def take_actions(self, actions):
        for id, agent in self.agents.items():
            if agent.predator:
                self._take_action(agent, actions[id])
                self.decrease_health(agent)
            else:
                self._take_action(agent, actions[id])

    def _take_action(self, agent, action):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w) and self.map[x][y] == 0
        x, y = agent.pos
        if action == 0:
            new_x = x - 1
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_x < 0:
                new_x = self.h-1
                new_y = y
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 1:
            new_x = x + 1
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_x >= self.h:
                new_x = 0
                new_y = y
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 2:
            new_x = x
            new_y = y - 1
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_y < 0:
                new_x = x
                new_y = self.w-1
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 3:
            new_x = x
            new_y = y + 1
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_y >= self.w:
                new_y = 0
                new_x = x
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        else:
            print('Wrong action id')

        new_x, new_y = agent.pos
        self.map[x][y] = 0
        self.map[new_x][new_y] = agent.id



        ## Exclude Grouping

    def decrease_health(self, agent):
        #for i in range(self.predator_num):
        #for i in range(len(self.agents)):
            #self.agents[i].health -= self.args.damage_per_step
        agent.health -= self.args.damage_per_step

    def increase_health(self, agent):
        if hasattr(self.args, 'health_increase_rate') and self.args.health_increase_rate is not None:
            agent.health += self.args.health_increase_rate
        else:
            agent.health += 1.


    def dump_image(self, img_name):
        new_w, new_h = self.w * 5, self.h * 5
        img = np.zeros((new_w, new_h, 3), dtype=np.uint8)
        length = self.args.img_length
        for i in range(self.h):
            for j in range(self.w):
                id = self.map[i][j]
                if self.food_map[i][j] == -2: img[(i*length-1):(i+1)*length, (j*length-1):(j+1)*length, :] = 255*np.array(self.property[-2][1])
                elif id == 0:
                    img[(i*length-1):(i+1)*length, (j*length-1):(j+1)*length, :] = 255
                elif id == -1:
                    img[(i*length-1):(i+1)*length, (j*length-1):(j+1)*length, :] = 255*np.array(self.property[id][1])
                else:
                    # prey
                    agent = self.agents[id]
                    img[(i*length-1):(i+1)*length, (j*length-1):(j+1)*length, :] = 255*np.array(agent.property[1])

        #for predator in self.predators.values():
        #    x, y = predator.pos
        #    img[(x*length-1):(x+1)*length, (y*length-1):(y+1)*length, :] = 255 * np.array(predator.property[1])
        output_img = Image.fromarray(img, 'RGB')
        output_img.save(img_name)

    def convert_img(self):
        img = np.zeros((self.h, self.w, 3))
        for i in range(self.h):
            for j in range(self.w):
                id = self.map[i][j]
                if self.food_map[i][j] == -2:
                    img[i, j, :] = 255*np.array(self.property[-2][1])
                elif id == 0:
                    img[i, j, :] = 255
                elif id == -1:
                    img[i, j, :] = 255*np.array(self.property[id][1])
                else:
                    # prey
                    img[i, j, :] = 255*np.array(self.property[-3][1])

        for predator in self.predators.values():
            x, y = predator.pos
            img[x, y, :] = 255*np.array(predator.property[1])
        return img



    def get_predator_reward(self, agent):
        reward = 0
        x, y = agent.pos
        min_dist = np.inf
        target_prey = None
        killed_id = None

        hunt_x = x-agent.hunt_square//2
        hunt_y = y-agent.hunt_square//2

        if hunt_x < 0:
            hunt_x = self.w+hunt_x
        if hunt_y < 0:
            hunt_y = self.h+hunt_y

        for i in range(agent.hunt_square):
            for j in range(agent.hunt_square):
                x_coord = hunt_x + i
                y_coord = hunt_y + j
                if x_coord < 0:
                    x_coord = self.w+x_coord
                if y_coord < 0:
                    y_coord = self.h+y_coord

                if x_coord >= self.w:
                    x_coord = x_coord - self.w
                if y_coord >= self.h:
                    y_coord = y_coord - self.h
                if self.map[x_coord][y_coord] > 0:
                    candidate_agent = self.agents[self.map[x_coord, y_coord]]
                    if not candidate_agent.predator:
                        x_prey, y_prey = candidate_agent.pos
                        dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
                        if dist < min_dist:
                            min_dist = dist
                            target_prey = candidate_agent
        if target_prey is not None:
            reward += 1
            target_prey.dead = True
            agent.max_reward += 1
            killed_id = target_prey.id
        #else:
        #    reward -= 1
        if agent.health <= 0:
            reward -= 1

        if agent.crossover:
            reward += 1
        return ((agent.id, reward), (agent.id, killed_id))

    def get_prey_reward(self, agent):
        reward = 0
        if agent.dead:
            reward -= 1
        #if not agent.dead:
        #    reward += 1

        if agent.crossover:
            reward += 1

        return ((agent.id, reward), (agent.id, None))

    def get_reward(self, agent):
        if agent.predator:
            #return self.get_predator_reward(agent) / len(self.predators)
            return self.get_predator_reward(agent)
        else:
            return self.get_prey_reward(agent)

    def remove_dead_agents(self):
        killed = []
        for agent in self.agents.values():
            #if agent.health <= 0 or np.random.rand() < 0.05:
            #if agent.health <= 0:
            if (agent.health <= 0):
                x, y = agent.pos
                self.map[x][y] = 0
                if agent.predator:
                    del self.predators[agent.id]
                    self.predator_num -= 1
                else:
                    del self.preys[agent.id]
                    self.prey_num -= 1
                killed.append(agent.id)
            elif agent.id in self.killed:
                # change this later
                killed.append(agent.id)
                del self.preys[agent.id]
                self.prey_num -= 1
                x, y = agent.pos
                self.map[x][y] = 0
            else:
                agent.age += 1
                agent.crossover=False
        self.killed = []
        return killed

    def _get_obs(self, agent):
        x, y = agent.pos
        obs = np.zeros((4+self.agent_emb_dim, self.vision_width, self.vision_height))
        obs[4:, self.vision_width//2, self.vision_height//2] = self.agent_embeddings[agent.id]
        vision_x = x-self.vision_width//2
        vision_y = y-self.vision_height//2

        if vision_x < 0:
            vision_x = self.w+vision_x
        if vision_y < 0:
            vision_y = self.h+vision_y

        for i in range(self.vision_width):
            for j in range(self.vision_height):
                x_coord = vision_x + i
                y_coord = vision_y + j
                if x_coord < 0:
                    x_coord = self.w+x_coord
                if y_coord < 0:
                    y_coord = self.h+y_coord

                if x_coord >= self.w:
                    x_coord = x_coord - self.w
                if y_coord >= self.h:
                    y_coord = y_coord - self.h

                if self.map[x_coord][y_coord] > 0:
                    other_agent = self.agents[self.map[x_coord][y_coord]]
                    obs[0, i, j] = other_agent.property[1][0]
                    obs[1, i, j] = other_agent.property[1][1]
                    obs[2, i, j] = other_agent.property[1][2]
                    obs[3, i, j] = other_agent.health
                elif self.map[x_coord][y_coord] == -1: #wall
                    obs[0, i, j] = 1.
                    obs[1, i, j] = 1.
                    obs[2, i, j] = 1.
                else:
                    obs[0, i, j] = self.property[0][1][0]
                    obs[1, i, j] = self.property[0][1][1]
                    obs[2, i, j] = self.property[0][1][2]



        if self.obs_type == 'dense':
            return (agent.id, obs[:4].reshape(-1))
        else:
            return (agent.id, obs)

    def _get_all(self, agent):
        x, y = agent.pos
        obs = np.zeros((4+self.agent_emb_dim, self.vision_width, self.vision_height))
        obs[4:, self.vision_width//2, self.vision_height//2] = self.agent_embeddings[agent.id]
        vision_x = x-self.vision_width//2
        vision_y = y-self.vision_height//2

        if vision_x < 0:
            vision_x = self.w+vision_x
        if vision_y < 0:
            vision_y = self.h+vision_y

        for i in range(self.vision_width):
            for j in range(self.vision_height):
                x_coord = vision_x + i
                y_coord = vision_y + j
                if x_coord < 0:
                    x_coord = self.w+x_coord
                if y_coord < 0:
                    y_coord = self.h+y_coord

                if x_coord >= self.w:
                    x_coord = x_coord - self.w
                if y_coord >= self.h:
                    y_coord = y_coord - self.h

                if self.map[x_coord][y_coord] > 0:
                    other_agent = self.agents[self.map[x_coord][y_coord]]
                    obs[:3, i, j] = other_agent.property[1]
                    obs[3, i, j] = other_agent.health
                elif self.map[x_coord][y_coord] == -1:
                    obs[:3, i, j] = 1.
                else:
                    obs[:3, i, j] = self.property[0][1]

        rewards, killed = self.get_reward(agent)

        if self.obs_type == 'dense':
            return (agent.id, obs[:4].reshape(-1)), rewards, killed
        else:
            return (agent.id, obs), rewards, killed

        if self.obs_type == 'dense':
            return (agent.id, obs[:4].reshape(-1)), rewards, killed
        else:
            return (agent.id, obs), rewards, killed

    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            del self.agent_embeddings[id]

    def render(self, only_view=False):
        if self.cpu_cores is None:
            cores = mp.cpu_count()
        else:
            cores = self.cpu_cores

        if self.args.multiprocessing:
            pool = mp.Pool(processes=cores)
            if only_view:
                obs = pool.map(self._get_obs, self.agents.values())
            else:
                obs = pool.map(self._get_all, self.agents.values())
            pool.close()
            pool.join()
        else:
            obs = []
            if only_view:
                for agent in self.agents.values():
                    obs.append(self._get_obs(agent))
            else:
                for agent in self.agents.values():
                    obs.append(self._get_all(agent))
            self.agent_embeddings = {}
        return obs

    def reset(self):
        self.__init__(self.args)
        self.agent_embeddings = {}
        self.make_world(wall_prob=self.args.wall_prob, wall_seed=self.args.wall_seed, food_prob=self.args.food_prob)

        return self.render

    def step(self, actions):
        self.timestep += 1
        self.take_actions(actions)
        rewards = {}

        obs, rewards, killed = zip(*self.render())
        self.killed = list(dict(killed).values())


        for id, killed in killed:
            if killed is not None:
                self.increase_health(self.agents[id])
        return obs, dict(rewards)


def get_obs(env, only_view=False):
    global agent_emb_dim
    agent_emb_dim = env.agent_emb_dim
    global vision_width
    vision_width = env.vision_width
    global vision_height
    vision_height = env.vision_height
    global agent_embeddings
    agent_embeddings = env.agent_embeddings
    global agents
    agents = env.agents
    global cpu_cores
    cpu_cores = env.cpu_cores
    global h
    h = env.h
    global w
    w = env.w
    global _map
    _map = env.map
    global _property
    _property = env.property
    global obs_type
    obs_type = env.obs_type
    global get_closer_reward
    get_closer_reward = env.get_closer_reward


    if env.cpu_cores is None:
        cores = mp.cpu_count()
    else:
        cores = cpu_cores

    if env.args.multiprocessing:
        pool = mp.Pool(processes=cores)
        if only_view:
            obs = pool.map(_get_obs, agents.values())
            pool.close()
            pool.join()
            return obs
        else:
            obs = pool.map(_get_all, agents.values())
        pool.close()
        pool.join()
    else:
        obs = []
        if only_view:
            for agent in agents.values():
                obs.append(_get_obs(agent))
            return obs
        else:
            for agent in agents.values():
                obs.append(_get_all(agent))

    obs, rewards, killed = zip(*obs)
    #killed = list(dict(killed).values())
    for id, killed_agent in killed:
        if killed_agent is not None:
            env.increase_health(agents[id])
    killed = list(dict(killed).values())

    return obs, dict(rewards), killed



def _get_all(agent):
    x, y = agent.pos
    obs = np.zeros((4+agent_emb_dim, vision_width, vision_height))
    obs[4:, vision_width//2, vision_height//2] = agent_embeddings[agent.id]
    vision_x = x-vision_width//2
    vision_y = y-vision_height//2

    if vision_x < 0:
        vision_x = w+vision_x
    if vision_y < 0:
        vision_y = h+vision_y

    for i in range(vision_width):
        for j in range(vision_height):
            x_coord = vision_x + i
            y_coord = vision_y + j
            if x_coord < 0:
                x_coord = w+x_coord
            if y_coord < 0:
                y_coord = h+y_coord

            if x_coord >= w:
                x_coord = x_coord - w
            if y_coord >= h:
                y_coord = y_coord - h

            if _map[x_coord][y_coord] > 0:
                other_agent = agents[_map[x_coord][y_coord]]
                obs[:3, i, j] = other_agent.property[1]
                obs[3, i, j] = other_agent.health
            elif _map[x_coord][y_coord] == -1:
                obs[:3, i, j] = 1.
            else:
                obs[:3, i, j] = _property[0][1]

    rewards, killed = _get_reward(agent)

    if obs_type == 'dense':
        return (agent.id, obs[:4].reshape(-1)), rewards, killed
    else:
        return (agent.id, obs), rewards, killed


def _get_obs(agent):
    x, y = agent.pos
    obs = np.zeros((4+agent_emb_dim, vision_width, vision_height))
    obs[4:, vision_width//2, vision_height//2] = agent_embeddings[agent.id]
    vision_x = x-vision_width//2
    vision_y = y-vision_height//2

    if vision_x < 0:
        vision_x = w+vision_x
    if vision_y < 0:
        vision_y = h+vision_y

    for i in range(vision_width):
        for j in range(vision_height):
            x_coord = vision_x + i
            y_coord = vision_y + j
            if x_coord < 0:
                x_coord = w+x_coord
            if y_coord < 0:
                y_coord = h+y_coord

            if x_coord >= w:
                x_coord = x_coord - w
            if y_coord >= h:
                y_coord = y_coord - h

            if _map[x_coord][y_coord] > 0:
                other_agent = agents[_map[x_coord][y_coord]]
                obs[:3, i, j] = other_agent.property[1]
                obs[3, i, j] = other_agent.health
            elif _map[x_coord][y_coord] == -1:
                obs[:3, i, j] = 1.
            else:
                obs[:3, i, j] = _property[0][1]

    if obs_type == 'dense':
        return (agent.id, obs[:4].reshape(-1))
    else:
        return (agent.id, obs)

def get_predator_reward(agent):
    reward = 0
    x, y = agent.pos
    min_dist = np.inf
    target_prey = None
    killed_id = None

    hunt_x = x-agent.hunt_square//2
    hunt_y = y-agent.hunt_square//2

    if hunt_x < 0:
        hunt_x = w+hunt_x
    if hunt_y < 0:
        hunt_y = h+hunt_y

    for i in range(agent.hunt_square):
        for j in range(agent.hunt_square):
            x_coord = hunt_x + i
            y_coord = hunt_y + j
            if x_coord < 0:
                x_coord = w+x_coord
            if y_coord < 0:
                y_coord = h+y_coord

            if x_coord >= w:
                x_coord = x_coord - w
            if y_coord >= h:
                y_coord = y_coord - h
            if _map[x_coord][y_coord] > 0:
                candidate_agent = agents[_map[x_coord, y_coord]]
                if not candidate_agent.predator:
                    x_prey, y_prey = candidate_agent.pos
                    dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
                    if dist < min_dist:
                        min_dist = dist
                        target_prey = candidate_agent

    if target_prey is not None:
        reward += 1
        target_prey.dead = True
        agent.max_reward += 1
        killed_id = target_prey.id
    #else:
    #    reward -= 0.2

    if agent.health <= 0:
        reward -= 1
    if agent.crossover:
        reward += 1
    elif agent.get_closer:
        reward += get_closer_reward
    return ((agent.id, reward), (agent.id, killed_id))

def get_prey_reward(agent):
    reward = 0
    if agent.dead:
        reward -= 1
    #else:
    #    reward += 0.2
    if agent.crossover:
        reward += 1
    elif agent.get_closer:
        reward += get_closer_reward

    return ((agent.id, reward), (agent.id, None))

def _get_reward(agent):
    if agent.predator:
        #return self.get_predator_reward(agent) / len(self.predators)
        return get_predator_reward(agent)
    else:
        return get_prey_reward(agent)
