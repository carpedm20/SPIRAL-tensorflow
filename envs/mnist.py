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
from .mypaint_utils import *
from .base import Environment


class MNIST(Environment):

    action_sizes = {
            'pressure': [4],
            'jump': [2],
            #'color': [4],
            #'size': [2],
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
        self.pressures = np.arange(0.8, 0, -0.2)
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
                self.screen_size, self.location_size, 0)

        self.ends = utils.uniform_locations(
                self.screen_size, self.location_size, 0)

    def reset(self):
        if self.conditional:
            self.random_target = self.get_random_target()
        else:
            self.random_target = None

        self.s = tiledsurface.Surface()
        self.s.flood_fill(0, 0, (255, 255, 255), (0, 0, 64, 64), 0, self.s)
        self.s.begin_atomic()

        self._step = 0
        return self.state, self.random_target

    def draw(self, ac, s=None, dtime=1):
        if s is None:
            s = self.s

        jump = 0
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
            elif name == 'jump':
                jump = value

        if 'color' in self.action_sizes:
            self.b.brushinfo.set_color_rgb(color)

        if jump:
            pressure = 0

        if False:
            self.b.stroke_to(
                    self.s.backend,
                    x, y,
                    pressure,
                    -0.25, 0.75,
                    dtime)
        else:
            self.curve(c_x, c_y, 0, 0, x, y)

        self.s.end_atomic()
        self.s.begin_atomic()

    
    # Throughout this module these conventions are used:
    # sx, sy = starting point
    # ex, ey = end point
    # kx, ky = curve point from last line
    # lx, ly = last point from InteractionMode update
    def curve(self, cx, cy, sx, sy, ex, ey):
        #entry_p, midpoint_p, junk, prange2, head, tail
        entry_p, midpoint_p, prange1, prange2, h, t = \
                0.1, 0.1, 0.1, 0.1, 0.0001, 0.0001

        points_in_curve = 100
        mx, my = midpoint(sx, sy, ex, ey)
        length, nx, ny = length_and_normal(mx, my, cx, cy)
        cx, cy = multiply_add(mx, my, nx, ny, length*2)
        x1, y1 = difference(sx, sy, cx, cy)
        x2, y2 = difference(cx, cy, ex, ey)
        head = points_in_curve * h
        head_range = int(head)+1
        tail = points_in_curve * t
        tail_range = int(tail)+1
        tail_length = points_in_curve - tail

        # Beginning
        px, py = point_on_curve_1(1, cx, cy, sx, sy, x1, y1, x2, y2)
        length, nx, ny = length_and_normal(sx, sy, px, py)
        bx, by = multiply_add(sx, sy, nx, ny, 0.25)
        self._stroke_to(bx, by, entry_p)
        pressure = abs(1/head * prange1 + entry_p)
        self._stroke_to(px, py, pressure)

        for i in xrange(2, head_range):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            pressure = abs(i/head * prange1 + entry_p)
            self._stroke_to(px, py, pressure)

        # Middle
        for i in xrange(head_range, tail_range):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            self._stroke_to(px, py, midpoint_p)

        # End
        for i in xrange(tail_range, points_in_curve+1):
            px, py = point_on_curve_1(i, cx, cy, sx, sy, x1, y1, x2, y2)
            pressure = abs((i-tail)/tail_length * prange2 + midpoint_p)
            self._stroke_to(px, py, pressure)

    def _stroke_to(self, x, y, pressure):
        duration = 0.001
        self.b.stroke_to(
                self.s.backend,
                x, y,
                pressure,
                0.0, 0.0,
                duration)

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
    import utils as ut
    from config import get_args

    args = get_args()
    ut.train.set_global_seed(args.seed)

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
