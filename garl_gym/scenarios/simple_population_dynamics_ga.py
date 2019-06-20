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
from scipy.stats import norm


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
        large_map = np.zeros((w*3, h*3), dtype=np.int32)
        for i in range(3):
            for j in range(3):
                large_map[w*i:w*(i+1), h*j:h*(j+1)] = self.map

        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0
        for predator in list(self.predators.values()):
            x, y = predator.pos
            local_map = large_map[(self.w+x-crossover_scope//2):(self.w+x-crossover_scope//2+crossover_scope), (self.h+y-crossover_scope//2):(self.h+y-crossover_scope//2+crossover_scope)]
            agent_indices = np.where(local_map > 0)

            if len(agent_indices[0]) == 0 or predator.crossover:
                continue

            for candidate_x, candidate_y in zip(agent_indices[0], agent_indices[1]):
                candidate_id = local_map[candidate_x, candidate_y]
                candidate_agent = self.agents[candidate_id]
                if candidate_agent.predator and not candidate_agent.crossover and predator.id != candidate_agent.id:
                    candidate_agent.get_closer = True
                    if np.random.rand() < crossover_rate:
                        #for i in range(np.random.randint(self.args.max_predator_offsprings)):
                        candidate_agent.crossover = True
                        predator.crossover = True
                        child = Agent()
                        child.id = self.max_id
                        self.max_id += 1
                        new_embedding = np.random.normal(size=[self.agent_emb_dim])
                        self.agent_embeddings[child.id] = new_embedding
                        child.spped = None
                        child.predator = True
                        child.health = 2
                        child.hunt_square = self.max_hunt_square
                        child.property = [self._gen_power(child.id), [0, 0, 1]]
                        x = ind[0][perm[index]]
                        y = ind[1][perm[index]]
                        index += 1
                        self.map[x][y] = child.id
                        child.pos = (x, y)
                        self.predators[child.id] = child
                        self.predator_num += 1
                        ### decrease health?
                        #candidate_agent.health -= 0.5
                        #predator.health -= -0.5

    def crossover_prey(self, crossover_scope=3, crossover_rate=0.001):
        large_map = np.zeros((w*3, h*3), dtype=np.int32)
        for i in range(3):
            for j in range(3):
                large_map[w*i:w*(i+1), h*j:h*(j+1)] = self.map

        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0
        for prey in list(self.preys.values()):
            x, y = prey.pos
            local_map = large_map[(self.w+x-crossover_scope//2):(self.w+x-crossover_scope//2+crossover_scope), (self.h+y-crossover_scope//2):(self.h+y-crossover_scope//2+crossover_scope)]
            agent_indices = np.where(local_map > 0)

            if len(agent_indices[0]) == 0 or prey.crossover:
                continue

            for candidate_x, candidate_y in zip(agent_indices[0], agent_indices[1]):
                candidate_id = local_map[candidate_x, candidate_y]
                candidate_agent = self.agents[candidate_id]

                if not candidate_agent.predator and not candidate_agent.crossover and candidate_agent.id != prey.id:
                    candidate_agent.get_closer = True
                    if np.random.rand() < crossover_rate:
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

    def increase_predator(self, prob):
        num = max(1, int(len(self.predators) * prob))
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))

        for i in range(num):
            agent = Agent()
            agent.health = 1
            agent.original_health = 1
            agent.birth_time = self.timestep
            agent.predator = True

            agent.id = self.max_id
            self.max_id += 1
            agent.speed = 1
            agent.hunt_square = self.max_hunt_square
            agent.property = [self._gen_power(agent.id), [0, 0, 1]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            if self.map[x][y] == 0:
                self.map[x][y] = agent.id
                agent.pos = (x, y)
            self.predators[agent.id] = agent
            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[agent.id] = new_embedding

    def increase_prey(self, prob):
        num = max(1, int(len(self.preys) * prob))
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        for i in range(num):
            agent = Agent()
            agent.health = 1
            agent.original_health = 1
            agent.birth_time = self.timestep
            agent.predator = False

            agent.id = self.max_id
            self.max_id += 1
            agent.property = [self._gen_power(agent.id), [1, 0, 0]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            if self.map[x][y] == 0:
                self.map[x][y] = agent.id
                agent.pos = (x, y)
            self.preys[agent.id] = agent
            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[agent.id] = new_embedding



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
                elif id <= 0 and id > -2:
                    img[i, j, :] = 255*np.array(self.property[id][1])
                else:
                    # prey
                    img[i, j, :] = 255*np.array(self.property[-3][1])

        for predator in self.predators.values():
            x, y = predator.pos
            img[x, y, :] = 255*np.array(predator.property[1])
        return img


    def remove_dead_agents(self):
        killed = []
        for agent in self.agents.values():
            #if agent.health <= 0 or np.random.rand() < 0.05:
            #if agent.health <= 0:
            death_prob = norm.cdf(agent.age, loc=400, scale=150)
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
            #elif not agent.predator and np.random.rand() < death_prob: # Change this later
            #    killed.append(agent.id)
            #    del self.preys[agent.id]
            #    self.prey_num -= 1
            #    x, y = agent.pos
            #    self.map[x][y] = 0
            else:
                agent.age += 1
                agent.crossover=False
        self.killed = []
        return killed

    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            del self.agent_embeddings[id]

    def reset(self):
        self.__init__(self.args)
        self.agent_embeddings = {}
        self.make_world(wall_prob=self.args.wall_prob, wall_seed=np.random.randint(5000), food_prob=self.args.food_prob)

        return get_obs(self, only_view=True)

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
    global large_map
    large_map = np.zeros((w*3, h*3), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            large_map[w*i:w*(i+1), h*j:h*(j+1)] = _map

    if env.cpu_cores is None:
        cores = mp.cpu_count()
    else:
        cores = cpu_cores

    if env.args.multiprocessing and len(agents)>6000:
        pool = mp.Pool(processes=cores)
        obs = pool.map(_get_obs, agents.values())
        pool.close()
        pool.join()
    else:
        obs = []
        for agent in agents.values():
            obs.append(_get_obs(agent))

    if only_view:
        return obs

    killed = []
    for agent in agents.values():
        killed.append(_get_killed(agent, killed))

    killed = dict(killed)

    global _killed
    _killed = killed

    if env.args.multiprocessing and len(agents)>6000:
        pool = mp.Pool(processes=cores)
        rewards = pool.map(_get_reward, agents.values())
        pool.close()
        pool.join()
    else:
        rewards = []
        for agent in agents.values():
            reward = _get_reward(agent)

    for id, killed_agent in killed.items():
        if killed_agent is not None:
            env.increase_health(agents[id])
    killed = list(killed.values())

    return obs, dict(rewards), killed



def _get_obs(agent):
    x, y = agent.pos
    obs = np.zeros((4+agent_emb_dim, vision_width, vision_height))
    obs[:3, :, :] = np.broadcast_to(np.array(_property[0][1]).reshape((3, 1, 1)), (3, vision_width, vision_height))
    obs[4:, vision_width//2, vision_height//2] = agent_embeddings[agent.id]
    local_map = large_map[(w+x-vision_width//2):(w+x-vision_width//2+vision_width), (h+y-vision_height//2):(h+y-vision_height//2+vision_height)]
    agent_indices = np.where(local_map!=0)
    if len(agent_indices[0]) == 0:
        if obs_type == 'dense':
            return (agent.id, obs[:4].reshape(-1))
        else:
            return (agent.id, obs)
    for other_x, other_y in zip(agent_indices[0], agent_indices[1]):
        id_ = local_map[other_x, other_y]

        if id_ == -1:
            obs[:3, other_x, other_y] = 1.
        else:
            other_agent = agents[local_map[other_x, other_y]]
            obs[:3, other_x, other_y] = other_agent.property[1]
            obs[3, other_x, other_y] = other_agent.health


    if obs_type == 'dense':
        return (agent.id, obs[:4].reshape(-1))
    else:
        return (agent.id, obs)

def _get_killed(agent, killed):
    if not agent.predator:
        return (agent.id, None)
    x, y = agent.pos
    min_dist = np.inf
    target_prey = None
    killed_id = None

    local_map = large_map[(w+x-agent.hunt_square//2):(w+x-agent.hunt_square//2+agent.hunt_square), (h+y-agent.hunt_square//2):(h+y-agent.hunt_square//2+agent.hunt_square)]
    agent_indices = np.where(local_map>0)

    if len(agent_indices[0]) == 0:
        return (agent.id, None)
    for candidate_x, candidate_y in zip(agent_indices[0], agent_indices[1]):
        id_ = local_map[candidate_x, candidate_y]
        candidate_agent = agents[id_]

        if not candidate_agent.predator and not candidate_agent.id in dict(killed).values():
            x_prey, y_prey = candidate_agent.pos
            dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
            if dist < min_dist:
                min_dist = dist
                target_prey = candidate_agent

    if target_prey is not None:
        killed_id = target_prey.id
    return (agent.id, killed_id)



def _get_reward(agent):
    reward = 0
    if agent.predator:
        if _killed[agent.id] is not None:
            reward += 1

        if agent.crossover:
            reward += 1

        if agent.health <= 0:
            reward -= 1
    else:
        if agent.id in _killed.values():
            reward -= 1
        if agent.crossover:
            reward += 1
        #else:
        #    reward += 0.2

    return (agent.id, reward)

