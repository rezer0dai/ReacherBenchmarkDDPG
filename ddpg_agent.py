import numpy as np
import random, copy, sys

import torch
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

from model import Critic#Actor,

from nes import *
from normalizer import *

sys.path.append("PrioritizedExperienceReplay")
from PrioritizedExperienceReplay.proportional import Experience as ReplayBuffer
# random memory is currently at least 3x faster
#from memory import Memory as ReplayBuffer

#with prio experience buffer we prefer larger buffer as we can sample efficiently
BUFFER_SIZE = int(1e6)#70 * 1000# last 70 eps is OK for brute-20-arms environment
BATCH_SIZE = 256        # minibatch size works well with 40:7 settings
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic

N_STEP = 3#1#
SOFT_SYNC = 7
RESAMPLE_DELAY = 3

PRIO_ALPHA = .8
PRIO_BETA = .9
PRIO_MIN = 1e-10
PRIO_MAX = 1e2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module): # share common preprocessing layer!
    def __init__(self, normalizer, state_size, action_size, random_seed, device):
        super().__init__()
        self.norm = normalizer(state_size)
        self.actor = NoisyNet([state_size, 400, 300, action_size]).to(device)
        self.critic = Critic(state_size, action_size, random_seed).to(device)

    def parameters(self):
        assert False, "should not be accessed!"

    def actor_parameters(self):
        return np.concatenate([
            list(self.norm.parameters()),
            self.actor.parameters()])

    def critic_parameters(self):
        return np.concatenate([
            list(self.norm.parameters()),
            list(self.critic.parameters())])

    def forward(self, states):
        states = torch.from_numpy(states).to(device)
        states = self.norm(states)
        pi = self.actor(states)
        pi = torch.tanh(pi)
        qa = self.critic(states, pi)
        return qa

    def act(self, states):
        states = torch.from_numpy(states).to(device)
        states = self.norm(states)
        pi = self.actor(states)
        pi = torch.tanh(pi)
        return pi

    def value(self, states, actions):
        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        states = self.norm(states)
        return self.critic(states, actions)

class Agent():
    def __init__(self, state_size, action_size, random_seed, learning_delay, learning_repeat, update_goal = None):
        self.learning_delay, self.learning_repeat = learning_delay, learning_repeat
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.n_discount = GAMMA ** N_STEP

        self.count = 0

        self.ac_explorer = ActorCritic(GlobalNormalizerWGrads, state_size, action_size, random_seed, device)
        self.ac_target = ActorCritic(GlobalNormalizerWGrads, state_size, action_size, random_seed, device)

        # sync
        self._soft_update(self.ac_target.actor, self.ac_explorer.actor, 1.)
        self._soft_update(self.ac_target.critic, self.ac_explorer.critic, 1.)

        # set optimizers, RMSprop is also good choice !
        self.actor_optimizer = optim.Adam(self.ac_explorer.actor_parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.ac_explorer.critic_parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, PRIO_ALPHA)
        self.update_goal = update_goal

    def step(self, state, action, reward, next_state, t):
        # append to prio buffer with maximum priority
        self.memory.add([state, action, reward, next_state], 1.)
        if len(self.memory) < BATCH_SIZE:
            return

        # skip x steps ~ give enough time to perform to get some feadback
        if 0 != t % self.learning_delay:
            return

        # postpone target + explorer sync
        self.count += 1
        tau = 0 if 0 != self.count % SOFT_SYNC else TAU

        # learn
        for _ in range(self.learning_repeat):
            batch, _, inds = self.memory.select(PRIO_BETA)
            td_errors = self.learn(batch, tau)
            self.memory.priority_update(inds, np.clip(np.abs(td_errors), PRIO_MIN, PRIO_MAX))

        # resampling noise not too often ~ keep geting used to noise
        if 0 != t % RESAMPLE_DELAY * self.learning_delay:
            return
        self.ac_explorer.actor.sample_noise()

    def explore(self, state): # exploration action
        action = self.ac_explorer.act(state).detach().cpu().numpy()
        return np.clip(action, -1, +1)

    def exploit(self, state): # exploitation action
        action = self.ac_target.act(state).detach().cpu().numpy()
        return np.clip(action, -1, +1)

    def _backprop(self, optim, loss, params):
        # learn
        optim.zero_grad() # scatter previous optimizer leftovers
        loss.backward() # propagate gradients
        torch.nn.utils.clip_grad_norm_(params, 1) # avoid (inf, nan) stuffs
        optim.step() # backprop trigger

    def learn(self, batch, tau):
        states, actions, rewards, n_states = zip(*batch)
        states, actions, rewards, n_states = np.vstack(states), np.vstack(actions), np.vstack(rewards), np.vstack(n_states)

        if None != self.update_goal:
            states, n_states, rewards = self.update_goal(
                    states, n_states, rewards, random.sample(
                        range(len(rewards)), random.randint(1, len(rewards) - 1)))

        # func approximators; self play
        n_qa = self.ac_target(n_states)
        # n_step target + bellman equation
        td_targets = rewards + self.n_discount * n_qa.detach()
        q_replay = self.ac_explorer.value(states, actions)
        critic_loss = F.mse_loss(q_replay, td_targets)#F.smooth_l1_loss(q_replay, td_targets)#
        self._backprop(self.critic_optimizer, critic_loss, self.ac_explorer.critic_parameters())

        # func approximators; self play
        qa = self.ac_explorer(states)
        # DDPG + td-lambda ( n-step error )
        td_error = qa - td_targets # w.r.t to self-played action from *0-step* state !!
        actor_loss = -td_error.mean()
        self._backprop(self.actor_optimizer, actor_loss, self.ac_explorer.actor_parameters())

        # sync by Soft-Actor-Critic approach
        self._soft_update(self.ac_explorer.actor, self.ac_target.actor, tau)
        self._soft_update(self.ac_explorer.critic, self.ac_target.critic, tau)

        return td_error.detach().cpu().numpy()

    def _soft_update(self, explorer, target, tau):
        if not tau:
            return

        for target_w, explorer_w in zip(target.parameters(), explorer.parameters()):
            target_w.data.copy_(
                target_w.data * (1. - tau) + explorer_w.data * tau)