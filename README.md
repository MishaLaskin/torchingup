# TorchingUp

# Why TorchingUp Exists

TorchingUp provides minimal implementations of common Reinforcement Learning algorithms written in PyTorch. It is meant to complement [OpenAI’s SpinningUp repository](https://github.com/openai/spinningup), which contains similar algorithms implemented in Tensorflow.

The repository is built with a pedagogical mindset - the point is to help you learn RL as efficiently as possible. If you’ve been following the [tutorials on SpinningUp](https://spinningup.openai.com/en/latest/), then TorchingUp is a natural way to start implementing common RL algorithms from scratch on your own.

To keep you focused on what’s important - *learning* - we follow a set of code design principles (see below) that are consistent across the various algorithms.

# Installation

Installation is simple. The code is compatible with Python 3 and has minimal dependencies. Set up a virtual environment called `torchingup` with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or [virtualenv](https://virtualenv.pypa.io/en/stable/installation/), and run

 `pip install -r requirements.txt`

 to install the dependencies. If you don’t have PyTorch installed, [you’ll need to do that first](https://pytorch.org/get-started/locally/). Also, note that the MuJoCo environments such as `InvertedPendulum`  require you to have a [MuJoCo license](https://www.roboti.us/license.html). If you’re a student, it’s free. Otherwise, you can sign up for the free trial. If you don’t have a license, that’s OK too - just use a different gym environment.

 Once you have everything set up, you can run

 `python algos/{algo_name}/{algo_name}.py`

 to run the code. For example, if you’re running the DQN, type

 `python algos/dqn/dqn.py`

 in your command line.


# What’s Included

Currently TorchingUp supports the following algorithms


1. Deep Q Network (Off-Policy)
2. Deep Deterministic Policy Gradient (Off-Policy)
3. REINFORCE (On-Policy)
4. Vanilla Policy Gradient (On-Policy)
5. Proximal Policy Optimization (On-Policy)

With support for TD3, ACER, Soft Actor Critic, and Hindsight Experience Replay coming up.

# Code Design

Reinforcement Learning (and more generally Deep Learning) algorithms are easily susceptible to bugs. The most common type of bug is silent - your code runs but the agent doesn’t learn anything. You then spend hours or days backtracking to find the mistake. To minimize these types of bug-hunting digressions, this repository structures code in a principled way.


## Split File Structure - `{algo_name}.py` and `core.py`

Each algorithm is split into an `{algo_name}.py` file (e.g. `dqn.py`  for DQNs) and `core.py` file. The code essential to understanding the algorithm is in the `{algo_name}` file while `core.py` contains auxiliary utility functions.


1. `{algo_name}.py` - code fundamental to understanding the algorithm. It contains the following structure:
  1. Initialization - initialize environment, networks, and replay buffer
  2. Gradient Update Rule - define a gradient update function (e.g.  `dqn_update()`). This is the *most important part of the algorithm* - it defines the optimization step.
  3. Training Loop - this loop collects experience, implements the update rule according to the algorithm, and logs the output at each epoch
2. `core.py` - contains utility classes and functions needed to implement the algorithm, such as:
  1. Neural Network classes (e.g. Q network, Policy network, Value network)
  2. Replay Buffer classes
  3. Environment Wrappers
  4. Loggers
  5. Miscellaneous utilities (e.g. hard and soft updates for off-policy target networks)


## Hyperparameters Stored in `config.json`

Since RL algorithms can have many hyperparameters, it’s important to keep them in one place separate from everything else. Each algorithm has an accompanying `config.json` file with hyperparameters. These are then loaded at the end of the `{algo_name}.py` file. When you play with hyperparameters, just edit the `config.json` file and run the algorithm.


## The Gradient Update Function

Each file has a standalone gradient update function. The reason is that the optimization step is the single most important piece of any RL algorithm. Pay close attention to what’s happening in this function to understand how the algorithm works.


## Modular, Shallow Classes

Note that all of the classes are modular and shallow. Instead implementing a master `Agent` class with tons of functionality, we keep our classes


1. Shallow - at most one level of inheritance and preferably no inheritance at all
2. Modular - each class achieves one thing

This practice keeps the code clean, minimal, with a lot more transparency when debugging than you’d get otherwise.


## Clear Documentation for Each Algorithm


Finally, each algorithm has a `README.md` file that provides a concise explanation of the algorithm. Every RL algorithm can be implemented in a number of ways that differ in subtle ways. Not knowing which version is being implemented is a common roadblock for efficient learning, which makes clear documentation an important part of any RL repository.

That’s about it!
Enjoy,
Misha Laskin
