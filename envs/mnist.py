from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# MyPaint
sys.path.append('libs/mypaint')
from lib import surface, tiledsurface, brush

from . import utils
from .base import Environment


class MNIST(Environment):

    action_sizes = {
            'pressure': [2],
            #'color': [4],
            'size': [2],
            'jump': [2],
            'control': None,
            'end': None,
    }

    mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')

    def __init__(self, args):
        super(MNIST, self).__init__(args)

        with open('assets/brushes/dry_brush.myb') as fp:
            self.bi = brush.BrushInfo(fp.read())
        self.b = brush.Brush(self.bi)

        self.background_color = (255, 255, 255)

        # jump
        self.jumps = [0, 1]

        # size
        self.sizes = np.arange(1, 0, -0.1)
        self.sizes = self.sizes * 1
        if 'size' in self.action_sizes:
            self.sizes = \
                    self.sizes[:self.action_sizes['size'][0]]

        # pressure
        self.pressures = np.arange(1.0, 0.9, -0.01)
        if 'pressure' in self.action_sizes:
            self.pressures = \
                    self.pressures[:self.action_sizes['pressure'][0]]

        self.colors = [
                (0., 0., 0.), # black
                (102., 217., 232.), # cyan 3
                (173., 181., 189.), # gray 5
                (255., 224., 102.), # yellow 3
        ]
        if 'color' in self.action_sizes:
            self.colors = self.colors[:self.action_sizes['color'][0]]
            self.colors = np.array(self.colors) / 255.

        self.controls = utils.uniform_locations(
                self.screen_size, self.location_size, 0,
                normalize=True)
        self.controls /= self.location_size

        self.ends = utils.uniform_locations(
                self.screen_size, self.location_size, 0)

    def reset(self):
        if self.conditional:
            self.random_target = self.get_random_target()
        else:
            self.random_target = None

        self.s = tiledsurface.Surface()
        self.s.flood_fill(0, 0, (255, 255, 255), (0, 0, 64, 64), 0, self.s)

        self._step = 0
        return self.state, self.random_target

    def draw(self, ac, s=None, dtime=0.05):
        if s is None:
            s = self.s

        x, y = self.ends[0]
        color = self.colors[0]
        pressure, size = self.pressures[0], self.sizes[0]

        for name in self.action_sizes:
            named_ac = ac[self.ac_idx[name]]
            value = getattr(self, name + "s")[named_ac]

            if name == 'end':
                x, y = value
            if name == 'control':
                c_x, c_y = value
            elif name == 'pressure':
                pressure = value
            elif name == 'size':
                size = value
            elif name == 'color':
                color = value
        
        self.s.begin_atomic()

        if 'color' in self.action_sizes:
            self.b.brushinfo.set_color_rgb(color)

        self.b.stroke_to(
                self.s.backend,
                x, y,
                pressure,
                c_x, c_y, dtime)

        self.s.end_atomic()
        self.s.begin_atomic()

    def get_random_target(self):
        return None

    def step(self, acs):
        self.draw(acs, self.s)
        self._step += 1
        terminal = (self._step == self.episode_length)
        if terminal:
            if self.conditional:
                reward = 1
                #reward = - utils.l2(self.state, self.random_target) \
                #        / np.prod(self.observation_shape) * 100
            else:
                reward = None
        else:
            reward = 0
        # state, reward, terminal, info
        return self.state, reward, terminal, {}

    def save_image(self, path):
        Image.fromarray(self.state).save(path)
        #self.s.save_as_png(path, alpha=False)

    @property
    def state(self):
        rect = [0, 0, self.height, self.width]
        scanline_strips = \
                surface.scanline_strips_iter(self.s, rect)
        data = next(scanline_strips)
        #data[:,:,:3][(255 == data[:,:,3])] = [255, 255, 255]
        return data

    def get_action_desc(self, ac):
        desc = []
        for name in self.action_sizes:
            named_ac = ac[self.ac_idx[name]]
            actual_ac = getattr(self, name+"s")[named_ac]
            desc.append("{}: {} ({})".format(name, actual_ac, named_ac))
        return "\n".join(desc)


class SimpleMNIST(MNIST):

    def __init__(self, args):
        super(SimpleMNIST, self).__init__(args)


if __name__ == '__main__':
    from config import get_args
    args = get_args()

    env = SimpleMNIST(args)

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print("[Step {}] ac: {}".format(
                    step, env.get_action_desc(action)))
            state, reward, terminal, info = env.step(action)
            step += 1
            
            if terminal:
                print("Ep #{} finished.".format(ep_idx))
                env.save_image("mnist{}.png".format(ep_idx))
                break
