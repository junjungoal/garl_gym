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
import multiprocessing as mp
import gc
from garl_gym.core import DiscreteWorld, Agent


class SimplePopulationDynamics(BaseEnv):
    '''
    Args:
        args(dict):  A dictionary of parameters such as height, width, and predator_num
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


        self.max_hunt_square = args.max_hunt_square
        self.max_speed = args.max_speed
        self.timestep = 0
        self.num_food = 0
        self.predator_id = 0
        self.prey_id = 0

        self.obs_type = args.obs_type

        self.agent_emb_dim = args.agent_emb_dim

        self.cpu_cores = args.cpu_cores

        self.increase_predators = 0
        self.increase_preys = 0
        self.large_map = np.zeros((self.w*3, self.h*3), dtype=np.int32)

        if hasattr(args, 'experiment_type'):
            self.experiment_type = args.experiment_type
        else:
            self.experiment_type = None


    @property
    def predator_agents(self):
        if self.experiment_type == 'variation':
            return {**self.random_predators, **self.trained_predators, **self.training_predators}
        else:
            return self.predators

    @property
    def prey_agents(self):
        if self.experiment_type == 'variation':
            return {**self.random_preys, **self.trained_preys, **self.training_preys}
        else:
            return self.preys

    @property
    def agents(self):
        if self.experiment_type == 'variation':
            return {**self.random_predators, **self.trained_predators, **self.training_predators, **self.random_preys, **self.trained_preys, **self.training_preys}
        else:
            return {**self.predators, **self.preys}

    @property
    def random_agents(self):
        return {**self.random_predators, **self.random_preys}

    @property
    def trained_agents(self):
        return {**self.trained_predators, **self.trained_preys}

    @property
    def training_agents(self):
        return {**self.training_predators, **self.training_preys}



    def gen_food(self, prob=0.1, seed=10):
        for i in range(self.h):
            for j in range(self.w):
                food_prob = np.random.rand()
                if food_prob < prob and self.map[i][j] != -1 and self.food_map[i][j] == 0:
                    self.food_map[i][j] = -2
                    self.num_food += 1

    def _init_property(self):
        self.property[-3] = [1, [1, 0, 0]]
        self.property[-2] = [1, [0, 1, 0]]
        self.property[-1] = [1, [0, 0, 0]]
        self.property[0] = [1, [0.411, 0.411, 0.411]]


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

    def increase_predator(self, prob):
        '''
        Generates new predators

        Args:
            prob: Ratio against the population which determins how many new agents generated.
        '''
        num = max(1, int(self.predator_num* prob))
        self.increase_predators = num

        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        if self.experiment_type == 'variation':
            total = len(self.random_predators) + len(self.trained_predators) + len(self.training_predators)
            p=[len(self.random_predators)/total, len(self.trained_predators)/total, len(self.training_predators)/total]

        for i in range(num):
            agent = Agent()
            agent.health = np.random.uniform(self.min_health, self.max_health)
            agent.original_health = agent.health
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
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = agent.id
                agent.pos = (x, y)
                if self.experiment_type == 'variation':
                    exp_type = np.random.choice(3, p=p)
                    if exp_type == 0:
                        agent.policy_type = 'random'
                        self.random_predators[agent.id] = agent
                    elif exp_type == 1:
                        agent.policy_type = 'trained'
                        self.trained_predators[agent.id] = agent
                    else:
                        agent.policy_type = 'training'
                        self.training_predators[agent.id] = agent
                else:
                    self.predators[agent.id] = agent
                self.predator_num += 1

    def increase_prey(self, prob):
        '''
        Generates new preys

        Args:
            prob: Ratio against the population which determins how many new agents generated.
        '''
        num = max(1, int(self.prey_num* prob))
        self.increase_preys = num
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        if self.experiment_type == 'variation':
            total = len(self.random_preys) + len(self.trained_preys) + len(self.training_preys)
            p=[len(self.random_preys)/total, len(self.trained_preys)/total, len(self.training_preys)/total]
        #for i in range(len(self.predator_agents)):
        for i in range(num):
         #   if np.random.rand() < prob:
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
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = agent.id
                agent.pos = (x, y)
                if self.experiment_type == 'variation':
                    exp_type = np.random.choice(3, p=p)
                    if exp_type == 0:
                        agent.policy_type = 'random'
                        self.random_preys[agent.id] = agent
                    elif exp_type == 1:
                        agent.policy_type = 'trained'
                        self.trained_preys[agent.id] = agent
                    else:
                        agent.policy_type = 'trainig'
                        self.training_preys[agent.id] = agent
                else:
                    self.preys[agent.id] = agent
                self.prey_num += 1

    def remove_dead_agents(self):
        '''
        Remove dead agents from the environment
        '''
        killed = []
        for agent in self.agents.values():
            #if agent.health <= 0 or np.random.rand() < 0.05:
            if agent.health <= 0:
            #if (agent.health <= 0 or agent.age >= agent.life):
                x, y = agent.pos
                self.map[x][y] = 0
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = 0
                if agent.predator:
                    if self.experiment_type == 'variation':
                        if agent.policy_type == 'random':
                            del self.random_predators[agent.id]
                        elif agent.policy_type == 'trained':
                            del self.trained_predators[agent.id]
                        else:
                            del self.training_predators[agent.id]
                    else:
                        del self.predators[agent.id]
                    self.predator_num -= 1
                else:
                    if self.experiment_type == 'variation':
                        if agent.policy_type == 'random':
                            del self.random_preys[agent.id]
                        elif agent.policy_type == 'trained':
                            del self.trained_preys[agent.id]
                        else:
                            del self.training_preys[agent.id]
                    else:
                        del self.preys[agent.id]
                    self.prey_num -= 1
                killed.append(agent.id)
            elif agent.id in self.killed:
                # change this later
                killed.append(agent.id)
                if self.experiment_type == 'variation':
                    if agent.policy_type == 'random':
                        del self.random_preys[agent.id]
                    elif agent.policy_type == 'trained':
                        del self.trained_preys[agent.id]
                    else:
                        del self.training_preys[agent.id]
                else:
                    del self.preys[agent.id]
                self.prey_num -= 1
                x, y = agent.pos
                self.map[x][y] = 0
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = 0
            else:
                agent.age += 1
                agent.crossover=False
                agent.checked = []
        self.killed = []
        self.increase_predators = 0
        self.increase_preys = 0
        return killed

    def reset(self):
        '''
        Reset the environment
        '''
        self.__init__(self.args)
        if self.experiment_type == 'variation':
            self.variation_make_world(wall_prob=self.args.wall_prob)
            return get_obs_with_variation(self, only_view=True)
        else:
            self.make_world(wall_prob=self.args.wall_prob, food_prob=self.args.food_prob)
            return get_obs(self, only_view=True)


def get_obs(env, only_view=False):
    '''
    Returns observations, rewards and Ids of killed agents

    Args:
        only_view (bool): If true, then return observations, rewards and ids of killed agents, otherwise only observations

    '''

    global agent_emb_dim
    agent_emb_dim = env.agent_emb_dim
    global vision_width
    vision_width = env.vision_width
    global vision_height
    vision_height = env.vision_height
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
    large_map = env.large_map

    if env.cpu_cores is None:
        cores = mp.cpu_count()
    else:
        cores = cpu_cores

    if env.args.multiprocessing and len(agents)>4000:
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

    if env.args.multiprocessing and len(agents)>4000:
        pool = mp.Pool(processes=cores)
        rewards = pool.map(_get_reward, agents.values())
        pool.close()
        pool.join()
    else:
        rewards = []
        for agent in agents.values():
            reward = _get_reward(agent)
            rewards.append(reward)

    for id, killed_agent in killed.items():
        if killed_agent is not None:
            env.increase_health(agents[id])
    killed = list(killed.values())

    return obs, dict(rewards), killed


def get_obs_with_variation(env, only_view=False):
    global agent_emb_dim
    agent_emb_dim = env.agent_emb_dim
    global vision_width
    vision_width = env.vision_width
    global vision_height
    vision_height = env.vision_height
    global agents
    agents = env.agents

    global random_agents
    random_agents = env.random_agents
    global trained_agents
    trained_agents = env.trained_agents
    global training_agents
    training_agents = env.training_agents

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
    large_map = env.large_map

    if env.cpu_cores is None:
        cores = mp.cpu_count()
    else:
        cores = cpu_cores

    if env.args.multiprocessing and len(agents)>6000:
        pool = mp.Pool(processes=cores)
        trained_obs = pool.map(_get_obs, trained_agents.values())
        training_obs = pool.map(_get_obs, training_agents.values())
        pool.close()
        pool.join()
    else:
        trained_obs = []
        training_obs = []
        for agent in trained_agents.values():
            trained_obs.append(_get_obs(agent))
        for agent in training_agents.values():
            training_obs.append(_get_obs(agent))

    if only_view:
        return (trained_obs, training_obs)

    killed = []
    for agent in agents.values():
        killed.append(_get_killed(agent, killed))
    killed = dict(killed)

    global _killed
    _killed = killed

    if env.args.multiprocessing and len(agents)>6000:
        pool = mp.Pool(processes=cores)
        rewards = pool.map(_get_reward, training_agents.values())
        pool.close()
        pool.join()
    else:
        rewards = []
        for agent in training_agents.values():
            reward = _get_reward(agent)
            rewards.append(reward)

    for id, killed_agent in killed.items():
        if killed_agent is not None:
            env.increase_health(agents[id])
    killed = list(killed.values())
    return (trained_obs, training_obs), dict(rewards), killed




def _get_obs(agent):
    x, y = agent.pos
    obs = np.zeros((4, vision_width, vision_height))
    obs[:3, :, :] = np.broadcast_to(np.array(_property[0][1]).reshape((3, 1, 1)), (3, vision_width, vision_height))
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
        return (agent.id, obs.reshape(-1))
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

        if not candidate_agent.predator and candidate_agent.id not in dict(killed).values():
            x_prey, y_prey = candidate_agent.pos
            dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
            if dist < min_dist:
                min_dist = dist
                target_prey = candidate_agent

    if target_prey is not None:
        killed_id = target_prey.id
        return (agent.id, killed_id)
    return (agent.id, killed_id)


def _get_reward(agent):
    if agent.predator:
        reward = 0
        if _killed[agent.id] is not None:
            reward += 1
        else:
            reward -= 0.001

        if agent.health <= 0:
            reward -= 4

    else:
        reward = 0
        if agent.id in _killed.values():
            reward -= 4
        else:
            reward += 0.001
        #else:
        #    reward += 0.2

    return (agent.id, reward)
