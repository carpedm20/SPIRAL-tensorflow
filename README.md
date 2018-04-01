# SPIRAL in TensorFlow (in progress)

TensorFlow implementation of [Synthesizing Programs for Images using Reinforced Adversarial Learning](https://deepmind.com/blog/learning-to-generate-images/) (**SPIRAL**).

![model](assets/model.png)

**SPIRAL** is an adversarially trained agent that generates a program which is executed by a graphics engine to interpret and sample images. This agent is trained to fool a discriminator with a distributed reinforcement learning without any supervision.

In short, Distributed RL + GAN + Program synthesis.


## Prerequisites

- *Python 2.7*
- [MyPaint 1.2.x](https://github.com/mypaint/mypaint/tree/v1.2.x)
- [TensorFlow 1.6.0](http://pytorch.org/)


## Usage

Install prerequisites with:

    ./install.sh
    pip install -r requirements.txt

To debug a **SPIARL** model:

    python run.py --num_workers 12 --env simple --episode_length=1 --loss=l2 --conditional=True

To train a **SPIARL** model:

    python run.py --num_workers 12 --env simple_mnist --episode_length=4 --loss=gan --conditional=True --location_size=32


## Results

![model](assets/mnist_in_progress1.png) ![model](assets/mnist_in_progress2.png)

(in progress)


## To-do

- [x] IMPALA A2C
- [x] Simple environment (debugging)
- [ ] Find a correct libmypaint setting
- [ ] MNIST environment
- [ ] ReplayThread (`--loss=gan`)
- [ ] Population Based Training (to be honest, I don't have any plan for this)


## References

*This code is heavily based on [openai/universe-starter-agent](https://github.com/openai/universe-starter-agent).*

- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
