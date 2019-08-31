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


class GeneticPopulationDynamics(BaseEnv):
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

        self.min_resilience = args.min_resilience
        self.max_resilience = args.max_resilience

        self.min_attack = args.min_attack
        self.max_attack = args.max_attack

        self.max_id = 1

        self.rewards = None

        self.max_view_size = None
        self.min_view_size = None
        self._init_property()

        self.get_closer_reward = args.get_closer_reward


        self.max_hunt_square = args.max_hunt_square
        self.max_speed = args.max_speed
        self.timestep = 0
        self.num_food = 0

        self.obs_type = args.obs_type

        self.agent_embeddings = {}
        self.agent_emb_dim = args.agent_emb_dim

        self.cpu_cores = args.cpu_cores

        self.increase_preys = 0
        self.increase_predators = 0
        self.large_map = np.zeros((self.w*3, self.h*3), dtype=np.int32)

        self.predator_health = []
        self.prey_health = []
        self.predator_attack = []
        self.prey_attack = []
        self.predator_resilience = []
        self.prey_resilience = []

        self.min_speed = args.min_speed
        self.max_speed = args.max_speed
        if hasattr(args, 'experiment_type'):
            self.experiment_type = args.experiment_type
        else:
            self.experiment_type = None



    #@property
    #def predators(self):
    #    return self.agents[:self.predator_num]

    #@property
    #def preys(self):
    #    return self.agents[self.predator_num:]

    @property
    def agents(self):
        return {**self.predators, **self.preys}



    def make_world(self, wall_prob=0, wall_seed=10):
        '''
        This function needs to be called at the initialisation
        Args:
            wall_prob: Probability of generating a wall on a cell
            wall_seed: Random seed for walls
        '''
        self.gen_wall(wall_prob, wall_seed)

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
            agent.life = np.random.normal(500, scale=100)
            agent.age = np.random.randint(150)

            agent.resilience = np.random.uniform(self.min_resilience, self.max_resilience)
            agent.gene_resilience = agent.resilience
            agent.attack = np.random.uniform(self.min_attack, self.max_attack)
            agent.gene_attack = agent.attack
            if i < self.predator_num:
                agent.predator = True
                agent.id = self.max_id
                agent.speed = np.random.randint(self.min_speed, self.max_speed)
                agent.gene_speed = agent.speed
                agent.hunt_square = self.max_hunt_square
                agent.property = [self._gen_power(i+1), [0, 0, 1]]
            else:
                agent.predator = False
                agent.id = self.max_id
                agent.property = [self._gen_power(i+1), [1, 0, 0]]
                agent.speed = np.random.randint(self.min_speed, self.max_speed)
                agent.gene_speed = agent.speed
            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[agent.id] = new_embedding

            x = empty_cells_ind[0][perm[i]]
            y = empty_cells_ind[1][perm[i]]
            self.map[x][y] = self.max_id
            agent.pos = (x, y)
            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = self.max_id
            self.max_id += 1

            if agent.predator:
                predators[agent.id] = agent
            else:
                preys[agent.id] = agent

            self.predators = predators
            self.preys = preys


    def gen_wall(self, prob=0, seed=10):
        if prob == 0:
            return
        #np.random.seed(seed)

        for i in range(self.h):
            for j in range(self.w):
                wall_prob = np.random.rand()
                buffer = []
                connected_wall = []
                if wall_prob < prob:
                    buffer.append((i, j))
                    connected_wall.append((i, j))

                    while len(buffer) != 0:
                        coord = buffer.pop()
                        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                if np.random.rand() < 0.15 and coord[0]+x>=0 and coord[0]+x<=self.h-1 and coord[1]+y>=0 and coord[1]+y<=self.w-1:
                                    buffer.append((coord[0]+x, coord[1]+y))
                                    connected_wall.append((coord[0]+x, coord[1]+y))
                                    self.map[coord[0]+x][coord[1]+y] = -1
                                    self.large_map[(coord[0]+x):self.large_map.shape[0]:self.map.shape[0], (coord[1]+y):self.large_map.shape[1]:self.map.shape[1]] = -1
                    if len(connected_wall) > 1:
                        for (x, y) in connected_wall:
                            self.map[x][y] = -1
                            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = -1

    def _init_property(self):
        self.property[-3] = [1, [1, 0, 0]]
        self.property[-2] = [1, [0, 1, 0]]
        self.property[-1] = [1, [0, 0, 0]]
        self.property[0] = [1, [0.411, 0.411, 0.411]]


    def crossover_predator(self, prob, mutation_prob=0.001):
        '''
        Crossover function for predators

        Args:
           prob: Ratio against the population. This determins how many agents are chosen for the crossover
           mutation_prob: Mutation probability
        '''

        num = max(1, int(self.predator_num* prob))
        self.increase_predators = num
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0

        predators = list(self.predators.values())
        np.random.shuffle(predators)

        for i in range(num):
            predator = predators[i*2]
            candidate_agent = predators[i*2+1]

            child = Agent()
            child.id = self.max_id
            self.max_id += 1
            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[child.id] = new_embedding
            child.life = np.random.normal(500, scale=100)
            child.predator = True

            rate = np.random.rand()
            child.health = np.random.uniform(self.min_health, self.max_health)

            rate = np.random.rand()
            if np.random.rand() < mutation_prob:
                child.attack = (rate*predator.gene_attack+(1-rate)*candidate_agent.gene_attack) + np.random.normal()
            else:
                child.attack = (rate*predator.gene_attack+(1-rate)*candidate_agent.gene_attack)

            child.resilience = 1.

            if np.random.rand() < mutation_prob:
                speed = (rate*predator.gene_speed+(1-rate)*candidate_agent.gene_speed) + np.random.normal()
                speed = np.clip(speed, self.min_speed, self.max_speed)
                child.speed = int(speed)
            else:
                child.speed = int(np.round(rate*predator.gene_speed+(1-rate)*candidate_agent.gene_speed))
            child.gene_speed = child.speed


            child.gene_attack = child.attack
            child.gene_resilience = child.resilience

            predator.reward = child.gene_attack + child.gene_resilience
            candidate_agent.reward = child.gene_attack + child.gene_resilience

            child.hunt_square = self.max_hunt_square
            child.property = [self._gen_power(child.id), [0, 0, 1]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            self.map[x][y] = child.id
            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = child.id
            child.pos = (x, y)
            self.predators[child.id] = child
            self.predator_num += 1

    def crossover_prey(self, prob, mutation_prob=0.001):
        '''
        Crossover function for preys

        Args:
           prob: Ratio against the population. This determins how many agents are chosen for the crossover
           mutation_prob: Mutation probability
        '''
        num = max(1, int(self.prey_num* prob))
        self.increase_preys = num
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        index = 0

        preys = list(self.preys.values())
        np.random.shuffle(preys)

        for i in range(num):
            prey = preys[i*2]
            candidate_agent = preys[i*2+1]
            child = Agent()
            child.id = self.max_id
            self.max_id += 1
            child.predator = False
            child.life = np.random.normal(500, scale=100)
            child.health = 1

            child.attack = 1

            rate = np.random.rand()
            if np.random.rand() < mutation_prob:
                child.resilience = (rate*prey.gene_resilience+(1-rate)*candidate_agent.gene_resilience) + np.random.normal()
            else:
                child.resilience = (rate*prey.gene_resilience+(1-rate)*candidate_agent.gene_resilience)

            rate = np.random.rand()
            if np.random.rand() < mutation_prob:
                speed = (rate*prey.gene_speed+(1-rate)*candidate_agent.gene_speed) + np.random.normal()
                speed = np.clip(speed, self.min_speed, self.max_speed)
                child.speed = int(speed)
            else:
                child.speed = int(np.round(rate*prey.gene_speed+(1-rate)*candidate_agent.gene_speed))

            child.gene_speed = child.speed
            child.gene_attack = child.attack
            child.gene_resilience = child.resilience
            child.gene_speed = child.speed
            prey.reward = child.gene_attack + child.gene_resilience
            candidate_agent.reward = child.gene_attack + child.gene_resilience

            new_embedding = np.random.normal(size=[self.agent_emb_dim])
            self.agent_embeddings[child.id] = new_embedding
            child.hunt_square = self.max_hunt_square
            child.property = [self._gen_power(child.id), [1, 0, 0]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            index += 1
            self.map[x][y] = child.id
            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = child.id
            child.pos = (x, y)
            self.preys[child.id] = child
            self.prey_num += 1


    def store_parameters(self, agent):
        if agent.predator:
            self.predator_health.append(agent.health)
            self.predator_attack.append(agent.gene_attack)
            self.predator_resilience.append(agent.gene_resilience)
            self.predator_speed.append(agent.gene_speed)

        else:
            self.prey_health.append(agent.health)
            self.prey_attack.append(agent.gene_attack)
            self.prey_resilience.append(agent.gene_resilience)
            self.prey_speed.append(agent.gene_speed)

    def reset_parameters(self):
        self.predator_health = []
        self.prey_health = []
        self.predator_attack = []
        self.prey_attack = []
        self.predator_resilience = []
        self.prey_resilience = []
        self.prey_speed = []
        self.predator_speed = []



    def remove_dead_agents(self):
        '''
        Remove dead agents from the environment
        '''

        killed = []
        self.reset_parameters()
        for agent in self.agents.values():
            #if agent.health <= 0 or np.random.rand() < 0.05:
            if agent.health <= 0:
            #if (agent.health <= 0 or agent.age >= agent.life):
                x, y = agent.pos
                self.map[x][y] = 0
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = 0
                if agent.predator:
                    del self.predators[agent.id]
                    self.predator_num -= 1
                    killed.append(agent.id)
                else:
                    if agentid in self.preys.keys():
                        del self.preys[agent.id]
                        self.prey_num -= 1
                        killed.append(agent.id)
            elif agent.predator and self.killed[agent.id] is not None and self.killed[agent.id] not in killed:
                prey_id = self.killed[agent.id]
                prey = self.agents[prey_id]
                predator = agent

                prey.resilience -= predator.attack
                if prey.resilience <= 0:
                    del self.preys[prey_id]
                    killed.append(prey_id)
                    self.prey_num -= 1
                    x, y = prey.pos
                    self.map[x][y] = 0
                    self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = 0
            else:
                self.store_parameters(agent)
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
        self.agent_embeddings = {}
        self.make_world(wall_prob=self.args.wall_prob, wall_seed=np.random.randint(5000))

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
    global agent_embeddings
    agent_embeddings = env.agent_embeddings
    global agents
    agents = env.agents
    global predators
    predators = env.predators
    global preys
    preys = env.preys
    global max_health
    max_health = env.max_health
    global max_resilience
    max_resilience = env.max_resilience
    global max_attack
    max_attack = env.max_attack
    global max_speed
    max_speed = env.max_speed


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

    #killed = []
    #for agent in agents.values():
    #    killed.append(_get_killed(agent))

    #killed = dict(killed)

    #global _killed
    #_killed = killed
    killed = []
    resiliences = {}
    for agent in predators.values():
        killed.append(_get_killed(agent, resiliences))
    killed = dict(killed)
    global _killed
    _killed = killed
    global killed_preys
    killed_preys = list(killed.values())
    global _resiliences
    _resiliences = resiliences


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
        if killed_agent is not None and _resiliences[_killed[id]] <= 0:
            env.increase_health(agents[id])
    #killed = list(killed.values())

    return obs, dict(rewards), killed



def _get_obs(agent):
    x, y = agent.pos
    obs = np.zeros((7, vision_width, vision_height))
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
            obs[3, other_x, other_y] = other_agent.health/max_health
            obs[4, other_x, other_y] = other_agent.attack / max_attack
            obs[5, other_x, other_y] = other_agent.resilience / max_resilience
            obs[6, other_x, other_y] = other_agent.speed / max_speed


    if obs_type == 'dense':
        return (agent.id, obs.reshape(-1))
    else:
        return (agent.id, obs)

def _get_killed(agent, resiliences):
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

        if not candidate_agent.predator:
            x_prey, y_prey = candidate_agent.pos
            dist = np.sqrt((x-x_prey)**2+(y-y_prey)**2)
            if dist < min_dist:
                min_dist = dist
                target_prey = candidate_agent

    if target_prey is not None:
        killed_id = target_prey.id
        if killed_id in resiliences.keys():
            resiliences[killed_id] -= agent.attack
        else:
            resiliences[killed_id] = target_prey.resilience-agent.attack
    return (agent.id, killed_id)


def _get_reward(agent):
    reward = 0
    if agent.predator:
        if _killed[agent.id] is not None and _resiliences[_killed[agent.id]] <= 0:
            num = killed_preys.count(_killed[agent.id])
            reward += 1./num

        if agent.crossover:
            reward += 1
            #reward += agent.reward

        if agent.health <= 0:
            reward -= 4

        if reward ==0:
            reward -= 0.001
    else:
        #if agent.id in _killed.values() or agent.health  <= 0:
        if (agent.id in _resiliences and _resiliences[agent.id] <= 0) or agent.health  <= 0:
            reward -= 4

        if agent.crossover:
           reward += 1

        if reward ==0:
            reward += 0.001
            #reward += agent.reward
        #    reward += 0.2

    return (agent.id, reward)


