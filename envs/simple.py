from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageDraw

from . import utils
from .base import Environment


class Simple(Environment):

    action_sizes = {
            'color': [1],
            'shape': [2],
            #'location': None,
    }

    def __init__(self, args):
        super(Simple, self).__init__(args)

        assert self.conditional, \
                "Don't train a Simple env with --conditional=False"

        # object
        self.object_radius = 3
        self.object_size = self.object_radius * 2
        self.background_color = (255, 255, 255)

        self.colors = [
                (0, 0, 0), # black
                (173, 181, 189), # gray 5
                (255, 224, 102), # yellow 3
                #(102, 217, 232), # cyan 3
        ]
        if 'color' in self.action_sizes:
            self.colors = self.colors[:self.action_sizes['color'][0]]

        self.shapes = ['circle', 'rectangle']
        if 'shape' in self.action_sizes:
            self.shapes = self.shapes[:self.action_sizes['shape'][0]]

        self.locations = utils.uniform_locations(
                self.screen_size, self.location_size,
                self.object_radius)

        self.image = None
        self.drawer = None
        self.random_target = None

    def reset(self):
        self.random_target = self.get_random_target()
        self.image = Image.new(
                'RGB', (self.width, self.height), self.background_color)
        self.drawer = ImageDraw.Draw(self.image)
        self._step = 0

        # TODO(taehoon): z
        self.z = None
        return self.state, self.random_target, self.z

    def draw(self, ac, drawer=None):
        if drawer is None:
            drawer = self.drawer

        r = self.object_radius
        x, y = self.locations[0]
        color, shape = self.colors[0], self.shapes[0]

        for name in self.action_sizes:
            named_ac = ac[self.ac_idx[name]]
            value = getattr(self, name + "s")[named_ac]

            if name == 'location':
                x, y = value
            elif name == 'color':
                color = value
            elif name == 'shape':
                shape = value

        if shape == 'circle':
            drawer.ellipse((x-r, y-r, x+r, y+r), fill=color)
        elif shape == 'rectangle':
            drawer.rectangle((x-r, y-r, x+r, y+r), fill=color)
        else:
            raise Exception("Unkown shape: {}".format(shape))

    def get_random_target(self):
        image = Image.new(
                'RGB', (self.width, self.height), self.background_color)
        drawer = ImageDraw.Draw(image)

        locations = []
        for _ in range(self.episode_length):
            ac = self.random_action(locations=locations)
            self.draw(ac, drawer)
            if 'location' in self.ac_idx:
                locations.append(ac[self.ac_idx['location']])
            else:
                locations.append(self.locations[0])

        return np.array(self.norm(image))

    def random_action(self, locations=[]):
        action = []
        for name  in self.acs:
            size = self.action_sizes[name]
            while True:
                sample = np.random.randint(np.prod(size))
                if name == 'locations':
                    if sample in locations:
                        continue
                else:
                    break
            action.append(sample)
        return action

    def step(self, acs):
        self.draw(acs, self.drawer)
        self._step += 1
        terminal = (self._step == self.episode_length)
        if terminal:
            if self.conditional:
                reward = - utils.l2(self.state, self.random_target) \
                        / np.prod(self.observation_shape) * 100
            else:
                reward = None
        else:
            reward = 0

        # XXX: DEBUG
        if reward == 0: reward = 1

        # state, reward, terminal, info
        return self.state, reward, terminal, {}

    def save_image(self, path):
        self.image.save(path)

    @property
    def state(self):
        return np.array(self.norm(self.image))


if __name__ == '__main__':
    from config import get_args
    args = get_args()

    env = Simple(args)

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print("[Step {}] ac: {}".format(step, action))
            state, reward, terminal, info = env.step(action)
            step += 1
            
            if terminal:
                print("Ep #{} finished.".format(ep_idx))
                env.save_image("simple{}.png".format(ep_idx))
                break
