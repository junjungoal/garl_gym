import os, sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import VideoWriter, imread, resize
import cv2

class BaseEnv(object):
    def make_world(self):
        raise NotImplementedError

    def reset_world(self):
        raise NotImplementedError

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

    def take_actions(self, actions):
        for id, action in actions.items():
            agent = self.agents[id]
            if agent.predator:
                self._take_action(agent, action)
                self.decrease_health(agent)
            else:
                self._take_action(agent, action)

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
