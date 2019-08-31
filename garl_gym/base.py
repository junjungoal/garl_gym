import os, sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import VideoWriter, imread, resize
from garl_gym.core import Agent
import cv2

class BaseEnv(object):
    """
    A parent class for environments
    """

    def variation_make_world(self, wall_prob=0):
        """
        Generates an environment for the experiment which veirfies if agents are trained or not.
        Three types of agnets (random policy, trained policy without any continual learning, trained policy with continual learning)
        The proportion of the initial population among three types of agnets are same

        Args:
            wall_prob: the probability of generating a wall block at the specific coordinate
        """
        self.gen_wall(wall_prob)

        random_predators = {}
        trained_predators = {}
        training_predators = {}
        random_preys = {}
        trained_preys = {}
        training_preys = {}

        agents = [Agent() for _ in range(self.predator_num + self.prey_num)]

        empty_cells_ind = np.where(self.map == 0)
        perm = np.random.permutation(range(len(empty_cells_ind[0])))

        random_predator_num = int(self.predator_num/3.)
        trained_predator_num = int(self.predator_num/3.)
        training_predator_num = int(self.predator_num-random_predator_num-trained_predator_num)

        random_prey_num = int(self.prey_num/3.)
        trained_prey_num = int(self.prey_num/3.)
        training_prey_num = int(self.prey_num-random_prey_num-trained_prey_num)


        for i, agent in enumerate(agents):
            health = np.random.uniform(self.min_health, self.max_health)
            agent.health = health
            agent.original_health = health
            agent.birth_time = self.timestep

            agent.life = np.random.normal(500, scale=100)
            agent.age = np.random.randint(150)
            if i < self.predator_num:
                agent.predator = True
                agent.id = self.max_id
                agent.speed = 1
                agent.hunt_square = self.max_hunt_square
                agent.property = [self._gen_power(i+1), [0, 0, 1]]
                if i < random_predator_num:
                    agent.policy_type = 'random'
                elif i >= random_predator_num and i < random_predator_num+trained_predator_num:
                    agent.policy_type = 'trained'
                else:
                    agent.policy_type = 'training'
            else:
                agent.predator = False
                agent.id = i+1
                agent.property = [self._gen_power(i+1), [1, 0, 0]]

                if i < random_prey_num + self.predator_num:
                    agent.policy_type = 'random'
                elif i >= random_prey_num+self.predator_num and i < random_prey_num+trained_prey_num+self.predator_num:
                    agent.policy_type = 'trained'
                else:
                    agent.policy_type = 'training'
            x = empty_cells_ind[0][perm[i]]
            y = empty_cells_ind[1][perm[i]]
            self.map[x][y] = self.max_id
            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = self.max_id
            agent.pos = (x, y)
            self.max_id += 1

            if agent.predator:
                if agent.policy_type == 'random':
                    random_predators[agent.id] = agent
                elif agent.policy_type == 'trained':
                    trained_predators[agent.id] = agent
                else:
                    training_predators[agent.id] = agent
            else:
                if agent.policy_type == 'random':
                    random_preys[agent.id] = agent
                elif agent.policy_type == 'trained':
                    trained_preys[agent.id] = agent
                else:
                    training_preys[agent.id] = agent
            self.random_predators = random_predators
            self.trained_predators = trained_predators
            self.training_predators = training_predators
            self.random_preys = random_preys
            self.trained_preys = trained_preys
            self.training_preys = training_preys


    def make_world(self, wall_prob=0, seed=100):
        """
        initialise an environment 
        """
        self.gen_wall(wall_prob)

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
            agent.age = np.random.randint(350)
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

            x = empty_cells_ind[0][perm[i]]
            y = empty_cells_ind[1][perm[i]]
            self.map[x][y] = self.max_id
            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = self.max_id
            agent.pos = (x, y)
            self.max_id += 1

            if agent.predator:
                predators[agent.id] = agent
            else:
                preys[agent.id] = agent

            self.predators = predators
            self.preys = preys

    def reset_world(self):
        raise NotImplementedError

    def dump_image(self, img_name):
        '''
        Save the full observation of the environment as an image
        '''
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

    def take_actions(self, actions):
        '''
        Take action for each agent

        Args:
            actions (dict): {id: action(int) for agent 1, id: action(int) for agent 2, ....id: action(int) for agnet N}
        '''

        for id, action in actions.items():
            agent = self.agents[id]
            if agent.predator:
                self._take_action(agent, action)
                self.decrease_health(agent)
            else:
                self._take_action(agent, action)
            agent.age += 1

    def _take_action(self, agent, action):
        def in_board(x, y):
            return not (x < 0 or x >= self.h or y < 0 or y >= self.w) and self.map[x][y] == 0
        x, y = agent.pos
        if action == 0:
            new_x = x - agent.speed
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_x < 0:
                new_x = self.w + (x-agent.speed)
                new_y = y
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 1:
            new_x = x + agent.speed
            new_y = y
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_x >= self.h:
                new_x = x+agent.speed-self.w
                new_y = y
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 2:
            new_x = x
            new_y = y - agent.speed
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_y < 0:
                new_x = x
                new_y = self.h + (y-agent.speed)
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        elif action == 3:
            new_x = x
            new_y = y + agent.speed
            if in_board(new_x, new_y):
                agent.pos = (new_x, new_y)
            elif new_y >= self.w:
                new_y = y+agent.speed-self.h
                new_x = x
                if in_board(new_x, new_y):
                    agent.pos = (new_x, new_y)
        else:
            print('Wrong action id')

        new_x, new_y = agent.pos
        self.map[x][y] = 0
        self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = 0
        self.large_map[new_x:self.large_map.shape[0]:self.map.shape[0], new_y:self.large_map.shape[1]:self.map.shape[1]] = agent.id
        self.map[new_x][new_y] = agent.id

    def decrease_health(self, agent):
        '''
        decrease health of predator

        Args:
            agent (Agent): An instance of Agnet class
        '''
        #for i in range(self.predator_num):
        #for i in range(len(self.agents)):
            #self.agents[i].health -= self.args.damage_per_step
        #agent.health -= self.args.damage_per_step * agent.speed
        agent.health -= self.args.damage_per_step

    def increase_health(self, agent):
        '''
        increase health of predator

        Args:
            agent (Agent): An instance of Agent class
        '''
        agent = self.agents[agent.id]
        if hasattr(self.args, 'health_increase_rate') and self.args.health_increase_rate is not None:
            agent.health += self.args.health_increase_rate
        else:
            agent.health += 1.

    def plot_map_cv2(self, resize_width=1000, resize_height=1000):
        img = cv2.resize(self.convert_img(), (resize_width,resize_height),
                         interpolation=cv2.INTER_AREA)
        cv2.imshow("World2", img)

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

    def make_video(self, images, outvid=None, fps=5, size=None, is_color=True, format="XVID"):
        """
        Create a video from a list of images.
        @param      outvid      output video
        @param      images      list of images to use in the video
        @param      fps         frame per second
        @param      size        size of each frame
        @param      is_color    color
        @param      format      see http://www.fourcc.org/codecs.php
        """
        # fourcc = VideoWriter_fourcc(*format)
        # For opencv2 and opencv3:
        if int(cv2.__version__[0]) > 2:
            fourcc = cv2.VideoWriter_fourcc(*format)
        else:
            fourcc = cv2.cv.CV_FOURCC(*format)
        vid = None
        for image in images:
            assert os.path.exists(image)
            img = imread(image)
            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()

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

    def gen_wall(self, prob=0, seed=10):
        '''
        Generates walls

        Args:
            prob: The probability of geenrating a wall on a cell
            seed: random seed
        '''

        if prob == 0:
            return

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
                                    self.large_map[(coord[0]+x):self.large_map.shape[0]:self.map.shape[0], (coord[1]+y):self.large_map.shape[1]:self.map.shape[1]] = -1
                    if len(connected_wall) > 1:
                        for (x, y) in connected_wall:
                            self.map[x][y] = -1
                            self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = -1



    def add_predators(self, num):
        '''
        Add predators

        Args:
            num: Number of predator this function generates
        '''

        self.increase_predators += num
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))

        if self.experiment_type == 'variation':
            total = len(self.random_predators) + len(self.trained_predators) + len(self.training_predators)
            p=[len(self.random_predators)/total, len(self.trained_predators)/total, len(self.training_predators)/total]

        for i in range(num):
            agent = Agent()
            agent.health = 1
            agent.original_health = 1
            agent.birth_time = self.timestep
            agent.predator = True

            agent.id = self.max_id
            self.max_id += 1
            agent.speed = 1
            agent.life = np.random.normal(500, 100)
            agent.hunt_square = self.max_hunt_square
            agent.property = [self._gen_power(agent.id), [0, 0, 1]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            if self.map[x][y] == 0:
                self.map[x][y] = agent.id
                self.large_map[x:self.large_map.shape[0]:self.map.shape[0], y:self.large_map.shape[1]:self.map.shape[1]] = agent.id
                agent.pos = (x, y)
                self.predator_num += 1
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

    def add_preys(self, num):
        '''
        Add preys

        Args:
            num: Number of prey this function generates
        '''
        self.increase_preys += num
        ind = np.where(self.map == 0)
        perm = np.random.permutation(np.arange(len(ind[0])))
        if self.experiment_type == 'variation':
            total = len(self.random_preys) + len(self.trained_preys) + len(self.training_preys)
            p=[len(self.random_preys)/total, len(self.trained_preys)/total, len(self.training_preys)/total]
        for i in range(num):
            agent = Agent()
            agent.health = 1
            agent.original_health = 1
            agent.birth_time = self.timestep
            agent.predator = False
            agent.life = np.random.normal(500, 100)

            agent.id = self.max_id
            self.max_id += 1
            agent.property = [self._gen_power(agent.id), [1, 0, 0]]
            x = ind[0][perm[i]]
            y = ind[1][perm[i]]
            if self.map[x][y] == 0:
                self.map[x][y] = agent.id
                self.prey_num += 1
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
                        agent.policy_type = 'training'
                        self.training_preys[agent.id] = agent
                else:
                    self.preys[agent.id] = agent

    def return_total_reward(self, rewards):
        total_pred_reward = 0
        total_prey_reward = 0
        for id, reward in rewards.items():
            if self.agents[id].predator:
                total_pred_reward += reward
            else:
                total_prey_reward += reward
        if self.experiment_type == 'variation':
            return total_pred_reward / (len(self.trained_predators)+1e-8), total_prey_reward / (len(self.trained_preys)+1e-8)
        else:
            return total_pred_reward / len(self.predators), total_prey_reward / len(self.preys)

