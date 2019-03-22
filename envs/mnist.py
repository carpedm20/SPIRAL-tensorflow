from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from PIL import Image, ImageDraw
from collections import defaultdict

# MyPaint
sys.path.append('libs/mypaint')
from lib import surface, tiledsurface, brush

import utils as ut
from . import utils
from .mypaint_utils import *
from .base import Environment


class MNIST(Environment):
    head = 0.25
    tail = 0.75

    action_sizes = {
            'pressure': [2],
            'jump': [2],
            'size': [2],
            'control': None,
            'end': None,
    }

    size = 0.2
    pressure = 0.3

    def __init__(self, args):
        super(MNIST, self).__init__(args)
        self.mnist_nums = args.mnist_nums
        self.colorize = not args.train

        self.prepare_mnist()

        # jump
        self.jumps = [0, 1]

        # size
        self.sizes = np.arange(0.2, 2.0, 0.5)
        self.sizes = self.sizes * 1
        if 'size' in self.action_sizes:
            self.sizes = \
                    self.sizes[:self.action_sizes['size'][0]]

        # pressure
        self.pressures = np.arange(0.8, 0, -0.3)
        if 'pressure' in self.action_sizes:
            self.pressures = \
                    self.pressures[:self.action_sizes['pressure'][0]]

        self.colors = [
                (0., 0., 0.), # black
                (102., 217., 232.), # cyan 3
                (173., 181., 189.), # gray 5
                (255., 224., 102.), # yellow 3
                (229., 153., 247.), # grape 3
                (99., 230., 190.), # teal 3
                (255., 192., 120.), # orange 3
                (255., 168., 168.), # red 3
        ]
        self.colors = np.array(self.colors) / 255.

        if 'color' in self.action_sizes:
            self.colors = self.colors[:self.action_sizes['color'][0]]

        self.controls = utils.uniform_locations(
                self.screen_size, self.location_size, 0)

        self.ends = utils.uniform_locations(
                self.screen_size, self.location_size, 0)

    def reset(self):
        self.entry_pressure = np.min(self.pressures)

        if self.conditional:
            self.random_target = self.get_random_target(num=1, squeeze=True)
        else:
            self.random_target = None

        self.s = tiledsurface.Surface()
        self.s.flood_fill(0, 0, (255, 255, 255), (0, 0, 64, 64), 0, self.s)
        self.s.begin_atomic()

        with open(self.args.brush_path) as fp:
            self.bi = brush.BrushInfo(fp.read())
        self.b = brush.Brush(self.bi)

        self._step = 0
        self.s_x, self.s_y = None, None

        if self.args.conditional:
            self.z = None
        else:
            self.z = np.random.uniform(-0.5, 0.5, size=self.args.z_dim)

        return self.state, self.random_target, self.z

    def draw(self, ac, s=None, dtime=1):
        if s is None:
            s = self.s

        jump = 0
        x, y = self.ends[0]
        c_x, c_y = self.controls[0]
        color = self.colors[0]
        pressure, size = self.pressure, self.size

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
            elif name == 'jump':
                jump = value

        if self.colorize:
            self.b.brushinfo.set_color_rgb(self.colors[self._step])
        if 'size' in self.action_sizes:
            self.b.brushinfo.set_base_value('radius_logarithmic', size)

        if (self.s_x is None and self.s_y is None):
            # when self._step == 0
            pressure = 0
            self.s_x, self.s_y = 0, 0
            self._stroke_to(self.s_x, self.s_y, pressure)
        elif 'jump' in self.action_sizes and jump:
            pressure = 0
            self._stroke_to(self.s_x, self.s_y, pressure)
        else:
            self._stroke_to(self.s_x, self.s_y, pressure)

        self._draw(x, y, c_x, c_y, pressure, size, color, dtime)

    def _draw(self, x, y, c_x, c_y,
              pressure, size, color, dtime):
        end_pressure = pressure

        # if straight line or jump
        if 'control' not in self.action_sizes or pressure == 0:
            self.b.stroke_to(
                    self.s.backend, x, y, pressure, 0, 0, dtime)
        else:
            end_pressure = self.curve(
                    c_x, c_y, self.s_x, self.s_y, x, y, pressure)

        self.entry_pressure = end_pressure

        self.s_x, self.s_y = x, y

        self.s.end_atomic()
        self.s.begin_atomic()

    # sx, sy = starting point
    # ex, ey = end point
    # kx, ky = curve point from last line
    # lx, ly = last point from InteractionMode update
    def curve(self, cx, cy, sx, sy, ex, ey, pressure):
        #entry_p, midpoint_p, junk, prange2, head, tail
        entry_p, midpoint_p, prange1, prange2, h, t = \
                self._line_settings(pressure)

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

        return pressure

    def _stroke_to(self, x, y, pressure, duration=0.1):
        self.b.stroke_to(
                self.s.backend,
                x, y,
                pressure,
                0.0, 0.0,
                duration)

    def get_random_target(self, num=1, squeeze=False):
        random_idxes = np.random.choice(self.real_data.shape[0], num, replace=False)
        random_image = self.real_data[random_idxes]
        if squeeze:
            random_image = np.squeeze(random_image, 0)
        return random_image

    def step(self, acs):
        self.draw(acs, self.s)
        self._step += 1
        terminal = (self._step == self.episode_length)
        if terminal:
            if self.conditional:
                reward = 1
                reward += - utils.l2(self.state, self.random_target) \
                        / np.prod(self.observation_shape)
            else:
                reward = 0
        else:
            reward = 0
        # state, reward, terminal, info
        return self.state, reward, terminal, {}

    def save_image(self, path="test.png"):
        Image.fromarray(self.image.astype(np.uint8).squeeze()).save(path)
        #self.s.save_as_png(path, alpha=False)

    @property
    def image(self):
        rect = [0, 0, self.height, self.width]
        scanline_strips = \
                surface.scanline_strips_iter(self.s, rect)
        return next(scanline_strips)

    @property
    def state(self):
        return utils.rgb2gray(self.image)

    def get_action_desc(self, ac):
        desc = []
        for name in self.action_sizes:
            named_ac = ac[self.ac_idx[name]]
            actual_ac = getattr(self, name+"s")[named_ac]
            desc.append("{}: {} ({})".format(name, actual_ac, named_ac))
        return "\n".join(desc)

    def _line_settings(self, pressure):
        p1 = self.entry_pressure
        p2 = (self.entry_pressure + pressure) / 2
        p3 = pressure
        if self.head == 0.0001:
            p1 = p2
        prange1 = p2 - p1
        prange2 = p3 - p2
        return p1, p2, prange1, prange2, self.head, self.tail

    def prepare_mnist(self):
        ut.io.makedirs(self.args.data_dir)

        # ground truth MNIST data
        mnist_dir = self.args.data_dir / 'mnist'
        mnist = tf.contrib.learn.datasets.DATASETS['mnist'](str(mnist_dir))

        pkl_path = mnist_dir / 'mnist_dict.pkl'

        if pkl_path.exists():
            mnist_dict = ut.io.load_pickle(pkl_path)
        else:
            mnist_dict = defaultdict(lambda: defaultdict(list))
            for name in ['train', 'test', 'valid']:
                for num in self.args.mnist_nums:
                    filtered_data = \
                            mnist.train.images[mnist.train.labels == num]
                    filtered_data = \
                            np.reshape(filtered_data, [-1, 28, 28])

                    iterator = tqdm(filtered_data,
                                    desc="[{}] Processing {}".format(name, num))
                    for idx, image in enumerate(iterator):
                        # XXX: don't know which way would be the best
                        resized_image = ut.io.imresize(
                                image, [self.height, self.width],
                                interp='cubic')
                        mnist_dict[name][num].append(
                                np.expand_dims(resized_image, -1))
            ut.io.dump_pickle(pkl_path, mnist_dict)

        mnist_dict = mnist_dict['train' if self.args.train else 'test']

        data = []
        for num in self.args.mnist_nums:
            data.append(mnist_dict[int(num)])

        self.real_data = 255 - np.concatenate([d for d in data])


class SimpleMNIST(MNIST):

    action_sizes = {
            #'pressure': [2],
            'jump': [2],
            #'color': [4],
            #'size': [2],
            'control': None,
            'end': None,
    }

    def __init__(self, args):
        super(SimpleMNIST, self).__init__(args)


if __name__ == '__main__':
    import utils as ut
    from config import get_args

    args = get_args()
    ut.train.set_global_seed(args.seed)

    env = args.env.lower()

    if env == 'mnist':
        env = MNIST(args)
    elif env == 'simple_mnist':
        env = SimpleMNIST(args)
    else:
        raise Exception("Unkown environment: {}".format(args.env))

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print("[Step {}] ac: {}".format(
                    step, env.get_action_desc(action)))
            state, reward, terminal, info = env.step(action)
            env.save_image("mnist{}_{}.png".format(ep_idx, step))
            
            if terminal:
                print("Ep #{} finished ==> Reward: {}".format(ep_idx, reward))
                break

            step += 1
