import gym

import numpy as np
import pandas as pd
import wandb
import time
import cv2

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any
from collections import deque
from models import Model, ConvModel
import wandb

import argh
import sys



def img_display(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class FrameStackingAndResizingEnv:
    def __init__(self, env, w, h, num_stack=4):
        self.env = env
        self.n = num_stack
        self.w = w
        self.h = h

        self.buffer = np.zeros((num_stack, h, w), 'uint8')
        self.frame = None

    def _preprocess_frame(self, fram):
        image = cv2.resize(fram, (self.w, self.h))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def step(self, action):
        im, reward, done, info = self.env.step(action)
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        # 0,1,2 -> 1,2,3
        self.buffer[1:self.n, :, :] = self.buffer[0:self.n-1, :, :]
        self.buffer[0, :, :] = im
        return self.buffer.copy(), reward, done, info
    
    def render(self, mode):
        if mode == 'rgb_array':
            return self.frame
        return super(FrameStackingAndResizingEnv, self).render(mode)

    @property
    def observation_space(self):
        return np.zeros((self.n, self.h, self.w))

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        im = self.env.reset()
        self.frame = im.copy()
        im = self._preprocess_frame(im)
        self.buffer = np.stack([im]*self.n, 0)
        return self.buffer.copy()

    # def render(self, mode):
    #     self.env.render(mode)



@dataclass
class SARS:
    state: Any
    action: int
    reward: float
    done: bool
    next_state: Any


# class DQNAgent:
#     def __init__(self, model):
#         self.model = model

#     def get_actions(self, observations):
#         # obs shape is (N, 4)
#         # N is batch size
#         q_vals = self.model(observations)

#         # q_vals shape (N, 2)

#         # .max(-1) the last axis
#         return q_vals.max(-1)[1]


class ReplayBuffer():
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        # could possibly improve by making it a dque or a database
        # self.buffer = deque(maxlen=buffer_size)
        # even better...
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self, sars):
        # self.buffer.append(sars)
        # self.buffer = self.buffer[-self.buffer_size:]
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        # assert num_samples <= len(self.buffer)
        assert num_samples <= min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size:
            return random.sample(self.buffer[:self.idx], num_samples)
        return random.sample(self.buffer, num_samples)


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_step(model, state_transitions, tgt, num_actions, gamma, device):
    cur_states = torch.stack(([torch.Tensor(s.state)
                               for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward])
                            for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor(
        [1]) for s in state_transitions])).to(device)
    next_states = torch.stack(
        ([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    one_hot_actions = F.one_hot(
        torch.LongTensor(actions), num_actions).to(device)

    ## MSE Loss ##
    # loss = (((rewards + mask[:, 0] * (qvals_next*gamma) -
    #           torch.sum(qvals*one_hot_actions, -1))**2)).mean().to(device)
    ## Huber Loss ##
    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals*one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * (qvals_next*gamma))
    loss.backward()
    model.opt.step()
    return loss

def run_test_episode(model, env, device, max_steps=1000):
    frames = []
    obs = env.reset()
    frames.append(env.frame)

    idx = 0
    done = False
    reward = 0
    while not done and idx < max_steps:
        #TODO make it do non conv actions too
        action = model(torch.Tensor(obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        obs, r, done, _ = env.step(action)
        reward += r
        frames.append(env.frame)
    return reward, np.stack(frames, 0)



hyperparameter_defaults = dict(
    run_name=str(random.getrandbits(128)),
    env_name='CartPole-v0',
    max_reward=500,
    max_steps=5_000_000,
    memory_size=100_000,
    min_rb_size=10000,
    sample_size=2500,
    env_steps_before_train=100,
    tgt_model_update=5000,
    reward_scaler=100.0,
    eps_min=0.01,
    eps_decay=0.999999,
    gamma=0.99,
    learning_rate=0.0001
)

### Used to solve Cartpole ###

def dqnmain(project_name, do_boltzman_exploration=False, test=False, chkpt=None, hypeparams=hyperparameter_defaults, steps=1000, device='cuda'):
    image_arr = []

    if (not test):
        wdbrun = wandb.init( project=project_name, config=hypeparams, name=hypeparams['run_name'], reinit=True, monitor_gym=False)
        # run.save("*.pth")
        config = wdbrun.config
        max_reward = config.max_reward
        max_steps = config.max_steps
        memory_size = config.memory_size
        min_rb_size = config.min_rb_size
        sample_size = config.sample_size
        env_steps_before_train = config.env_steps_before_train
        tgt_model_update = config.tgt_model_update
        reward_scaler = config.reward_scaler
        eps_min = config.eps_min
        eps_decay = config.eps_decay
        gamma = config.gamma
        learning_rate = config.learning_rate
    else:
        max_reward = hypeparams['max_reward']
        max_steps = steps
        memory_size = hypeparams['memory_size']
        min_rb_size = hypeparams['min_rb_size']
        sample_size = hypeparams['sample_size']
        env_steps_before_train = hypeparams['env_steps_before_train']
        tgt_model_update = hypeparams['tgt_model_update']
        reward_scaler = hypeparams['reward_scaler']
        eps_min = hypeparams['eps_min']
        eps_decay = hypeparams['eps_decay']
        gamma = hypeparams['gamma']
        learning_rate = hypeparams['learning_rate']

    env = gym.make(hypeparams['env_name'])
    if hypeparams['env_name'] == 'Breakout-v0':
        #TODO
        env = FrameStackingAndResizingEnv(env, 84, 84, 4) # change stack size here
    env._max_episode_steps = 4000

    test_env = gym.make(hypeparams['env_name'])
    if hypeparams['env_name'] == 'Breakout-v0':
        #TODO
        test_env = FrameStackingAndResizingEnv(test_env, 84, 84, 4) # change stack size here
    test_env._max_episode_steps = 4000
    last_observation = env.reset()

    if hypeparams['env_name'] == 'Breakout-v0':
        m = ConvModel(env.observation_space.shape,
                env.action_space.n, learning_rate).to(device)
    else:
        m = Model(env.observation_space.shape,
                env.action_space.n, learning_rate).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))

    if hypeparams['env_name'] == 'Breakout-v0':
        tgt = ConvModel(env.observation_space.shape, env.action_space.n).to(
            device)
    else:
        tgt = Model(env.observation_space.shape, env.action_space.n).to(
            device)  # target model, gets update fewer times
    update_tgt_model(m, tgt)

    rb = ReplayBuffer(memory_size)
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1 * min_rb_size
    i = 0

    episode_rewards = []
    rolling_reward = 0
    solved = False

    try:
        while (not solved) and step_num < max_steps:
            if test:
                screen = env.render('rgb_array')
                image_arr.append(screen)
                eps = 0
            else:
                eps = eps_decay**(step_num)

            if do_boltzman_exploration:
                if hypeparams['env_name'] == 'Breakout-v0':
                        logits = m(torch.Tensor(last_observation).unsqueeze(0).to(device))[0]
                        action = torch.distributions.Categorical(logits=logits).sample().item()
                else:
                    logits = m(torch.Tensor(last_observation).to(device))[0]
                    action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    if hypeparams['env_name'] == 'Breakout-v0':
                        action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()
                    else:
                        action = m(torch.Tensor(last_observation).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            reward = reward/reward_scaler

            rb.insert(SARS(last_observation, action,
                           reward, done, observation))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                rolling_reward = 0
                observation = env.reset()

            steps_since_train += 1
            i += 1
            step_num += 1
            if (not test) and rb.idx > min_rb_size and steps_since_train > env_steps_before_train:
                loss = train_step(m, rb.sample(sample_size),
                                  tgt, env.action_space.n, gamma, device)
                ave_reward = np.mean(episode_rewards)
                wdbrun.log({'loss': loss.detach().cpu().item(
                ), 'epsilon': eps, 'avg_reward': ave_reward}, step=step_num)
                if ave_reward >= max_reward:
                    solved = True
                episode_rewards = []
                epochs_since_tgt += 1
                # print(step_num, loss.detach().item())
                if epochs_since_tgt > tgt_model_update:
                    # print('updating target model')
                    update_tgt_model(m, tgt)
                    rew, frames = run_test_episode(m, test_env, device)
                    # frames.shape == (T, H, W, C)
                    # wandb.log({'test_reward': rew, 'test_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    wandb.log({'test_reward': rew})
                    epochs_since_tgt = 0
                    torch.save(
                        tgt.state_dict(), f"{wandb.run.dir}/{hypeparams['run_name']}_{step_num}.pth")
                steps_since_train = 0
                if ave_reward >= max_reward:
                    solved = True
        wandb.join()
        env.close()
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    hyperset = dict(
    run_name=str(random.getrandbits(128)),
    env_name='Breakout-v0',
    max_reward=500,
    max_steps=2_000_000,
    memory_size=50_000,
    min_rb_size=20000,
    sample_size=128,
    env_steps_before_train=16,
    tgt_model_update=500,
    reward_scaler=1,
    eps_min=0.05,
    eps_decay=0.999999,
    gamma=0.99,
    learning_rate=0.0001
)
    # # argh.dispatch_command(dqnmain)
    # # # dqnmain()
    # env = gym.make("Breakout-v0")
    # env = FrameStackingAndResizingEnv(env, 480, 640)

    # # # print(env.observation_space.shape)

    # # # print(env.action_space)

    # im = env.reset()
    # idx = 0
    # ims = []
    # print(im.shape)
    # # for i in range(im.shape[-1]):
    # #     ims.append(im[:,:,i])
    # #     # cv2.imwrite(f"/tmp/{i}.jpg", im[:,:,i])
    # # img_display(np.hstack(ims))

    # env.step(1)

    # for _ in range(10):
    #     idx += 1
    #     im, _, _, _ = env.step(random.randint(0, 3))
    #     for i in range(im.shape[-1]):
    #         ims.append(im[:, :, i])
    #     img_display(np.hstack(ims))
    #     ims = []
if __name__ == '__main__':
    dqnmain('Breakout-Tutorial', do_boltzman_exploration=False, hypeparams=hyperset)