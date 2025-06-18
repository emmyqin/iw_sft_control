import os
import time
import gym
#Necessary Import for Gym.
import d4rl
import numpy as np
import torch
from copy import deepcopy
import wandb
import uuid
import argparse
from torch.distributions import Distribution

from buffer import EpisodicReplayBuffer
from net import GaussPolicyMLP
from torch.optim.lr_scheduler import _LRScheduler


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project='filtered_'+ config['env'],
        group='filtered_'+ config['env'],
        name='filtered_'+ config['env'],
        id=str(uuid.uuid4())
    )
    wandb.run.save()

class BCReplay:

        def __init__(self, device, states, actions, masks, returns):
            self._device = device
            self._episode_states = states
            self._episode_actions = actions
            self._episode_masks = masks
            self._num_episodes = self._episode_states.shape[0]
            self._return = returns

        @property
        def min_return(self):
            return np.min(self._return)

        @property
        def max_return(self):
            return np.max(self._return)
        
        def sample(self, bs):
            ind = np.random.randint(0, self._num_episodes, size=bs)
            states = self._episode_states[ind]
            actions = self._episode_actions[ind]
            masks = self._episode_masks[ind]
            ret = self._return[ind]
            ret = (ret - self.min_return) / self.max_return
            return (
                torch.FloatTensor(states).to(self._device), 
                torch.FloatTensor(actions).to(self._device), 
                torch.FloatTensor(masks).to(self._device),
                torch.FloatTensor(ret).to(self._device)
                )
        
        def filter_dataset(self, cutoff_return: float):
            idxs = np.argwhere(self._return[:, 0] > cutoff_return)[:, 0]
            self._episode_states = self._episode_states[idxs]
            self._episode_actions = self._episode_actions[idxs]
            self._episode_masks = self._episode_masks[idxs]
            self._return = self._return[idxs]
            self._num_episodes = len(idxs)
        
        def filter_dataset_via_length(self, cutoff: float):
            idxs = np.argwhere(self._episode_masks.sum(axis=-1) <= cutoff)[:, 0]
            self._episode_states = self._episode_states[idxs, :cutoff]
            self._episode_actions = self._episode_actions[idxs, :cutoff]
            self._episode_masks = self._episode_masks[idxs, :cutoff]
            self._return = self._return[idxs]
            self._num_episodes = len(idxs)


def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)
    


def filter_data_func(replay_buffer, cut_offs, device):
    strat_states = []
    strat_actions = []
    strat_masks = []
    strat_returns = []

    for fil in cut_offs:
        replay_buffer.filter_dataset(fil)
        strat_states.append(replay_buffer._episode_states)
        strat_actions.append(replay_buffer._episode_actions)
        strat_masks.append(replay_buffer._episode_masks)
        strat_returns.append(replay_buffer._return)

    strat_states = np.concatenate(strat_states, axis=0)
    strat_actions = np.concatenate(strat_actions, axis=0)
    strat_masks = np.concatenate(strat_masks, axis=0)
    strat_returns = np.concatenate(strat_returns, axis=0)
    filtered_bc_replay = BCReplay(device, strat_states, strat_actions, strat_masks, strat_returns) 
    return filtered_bc_replay


def log_prob_trajs(policy, states, actions, masks, sum=False):
    bs, eps_len = masks.shape
    batched_states = states.reshape(-1, states.shape[-1])
    batched_actions = actions.reshape(-1, actions.shape[-1])
    dist = policy(batched_states)
    log_probs = log_prob_func(dist, batched_actions)[..., 0]
    log_probs = log_probs.reshape(bs, eps_len) * masks
    if sum==True:
        return log_probs.sum(axis=-1)
    else:
        return log_probs

def loss_bc(batch, policy) -> torch.Tensor:
    states, actions, masks, _ = batch
    new_log_prob = log_prob_trajs(policy, states, actions, masks)
    loss = - new_log_prob.sum(axis=-1) / masks.sum(axis=-1)
    return loss.mean()

def loss(batch, policy, iw_policy, old_policy, temp=2., normalize=False, sum_traj=True):
    states, actions, masks, _ = batch

    new_log_prob = log_prob_trajs(policy, states, actions, masks, sum=sum_traj)
    
    with torch.no_grad():
        iw_log_prob = log_prob_trajs(iw_policy, states, actions, masks, sum=sum_traj)
        old_log_prob = log_prob_trajs(old_policy, states, actions, masks, sum=sum_traj)
        if sum_traj:
            diff_log = temp * (iw_log_prob - old_log_prob) / masks.sum(axis=-1)
            ratio = torch.clamp(diff_log.exp(), 0., 1000.).detach()
        else:
            diff_log = temp * (iw_log_prob - old_log_prob)
            ratio = torch.clamp(diff_log.exp(), 0.0, 2.).detach()
        if normalize:
            ratio = ratio / ratio.sum() 
    if sum_traj:
        loss =  - ratio.detach() * (new_log_prob / masks.sum(axis=-1))
    else:
        loss =  - (ratio.detach() * new_log_prob).sum(axis=-1) / masks.sum(axis=-1)
    return loss.mean(), ratio


def select_action(policy, s: torch.Tensor, is_sample: bool) -> torch.Tensor:
    dist = policy(s)
    if is_sample:
        action = dist.sample()
    else:    
        action = dist.mean
    return action

def evaluate(device, policy, env, seed: int, eval_episodes: int=10) -> float:
    env.seed(seed)

    total_reward = 0
    for _ in range(eval_episodes):
        s, done = env.reset(), False
        ep_reward = 0.
        while not done:
            s = torch.FloatTensor((np.array(s).reshape(1, -1))).to(device)
            a = select_action(policy, s, is_sample=False).cpu().data.numpy().flatten()
            s, r, done, _ = env.step(a)
            ep_reward += r
        total_reward += ep_reward
    
    avg_reward = total_reward / eval_episodes
    return avg_reward


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def ema_update(model1, model2, decay=0.999):
    if decay < 0.0 or decay > 1.0:
        raise ValueError(f"Invalid decay value {decay} provided. Please provide a value in [0,1] range.")

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        p1.data.copy_(decay * p1.data + (1. - decay) * p2.data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="walker2d-medium-replay-v2")        
    parser.add_argument("--path", default="logs")        
    parser.add_argument("--seed", default=8, type=int)         
   
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=10000, type=int) 
    parser.add_argument("--bc_hidden_dim", default=256, type=int)
    parser.add_argument("--bc_depth", default=3, type=int)
    parser.add_argument("--bc_lr", default=1e-3, type=float)
    parser.add_argument("--bc_batch_size", default=128, type=int)
    parser.add_argument("--bc_log_freq", default=10, type=int)
    # For Filtered BC
    parser.add_argument("--filter_warmup_steps", default=300, type=int)
    parser.add_argument("--filter_bc_steps", default=2000, type=int)
    parser.add_argument("--filter_lr", default=4e-5, type=float)  
    parser.add_argument("--filter_iw_lr", default=4e-5, type=float) 
    parser.add_argument("--filter_batch_size", default=256, type=int)
    parser.add_argument("--update_q", default=100, type=int)
    parser.add_argument("--update_q_ema_decay", default=0.9, type=float)
    parser.add_argument("--filter_log_freq", default=500, type=int) 
    parser.add_argument("--train_iw", default=True, type=bool) 
    parser.add_argument("--train_sft", default=False, type=bool) 
    parser.add_argument("--temp", default=10., type=float)  
    parser.add_argument("--sum_traj", default=True, type=bool)  
    parser.add_argument("--filter_via_length", default=False, type=bool)
    parser.add_argument("--normalize", default=True, type=bool) 
    parser.add_argument("--cut_off_percentile", default=[90., 95., 98.], nargs='+', type=float)
    
    args = parser.parse_args()
    print(f'------current env {args.env} and current seed {args.seed}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    wandb_init(config)
    env_name = args.env
    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    dataset = env.get_dataset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = EpisodicReplayBuffer('cuda', state_dim, action_dim, len(dataset['actions']))
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(1.)
    device = torch.device("cuda")
    ep_masks = torch.FloatTensor(replay_buffer._episode_masks).to(device)

    print(f'Using cut off percentile {args.cut_off_percentile}')
    perc_cut_offs = [np.percentile(replay_buffer._return, x) for x in args.cut_off_percentile]

    filtered_bc_replay = filter_data_func(
         replay_buffer=deepcopy(replay_buffer),
         cut_offs=perc_cut_offs,
         device=device
    )

    if args.filter_via_length:
        filtered_bc_replay.filter_dataset_via_length(1)
    print(f'Resulting number of trajectories: {filtered_bc_replay._num_episodes}')
    
    # Set up the BC Network.
    hidden_dim = args.bc_hidden_dim
    depth = args.bc_depth
    lr = args.bc_lr
    policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
    iw_policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device).requires_grad_(False)
    old_policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device).requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    bs = args.bc_batch_size
    iw_policy.load_state_dict(old_policy.state_dict())

    bc_checkpoint_path = env_name + f'_{args.bc_steps}.pt'
    
    if os.path.exists(bc_checkpoint_path):
        print(f'loading checkpoint from {bc_checkpoint_path}')
        policy.load_state_dict(torch.load(bc_checkpoint_path, map_location=device))
        step_offset = 0
    else:
        for i in range(args.bc_steps):
            batch = replay_buffer.sample(bs)
            policy_loss = loss_bc(batch, policy)
            wandb.log(dict(bc_loss=policy_loss))
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
          
            if i % args.bc_log_freq == 0:
                avg_reward = evaluate(device, policy, env, args.seed + i)
                d4rl_score = env.get_normalized_score(avg_reward) * 100
                wandb.log(dict(filter_rewards=d4rl_score, filter_rewards_iw=d4rl_score))
                print(f'----REWARD AT STEP {i}----{d4rl_score}, {policy_loss}')
        torch.save(policy.state_dict(), bc_checkpoint_path)
        step_offset = args.bc_steps
    
    
    # Set up filtered BC
    filter_lr = args.filter_lr
    filter_iw_lr = args.filter_iw_lr
    filter_bs = args.filter_batch_size
    update_q = args.update_q
    filter_bc_policy = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
    filter_bc_policy_iw = GaussPolicyMLP(state_dim, hidden_dim, depth, action_dim).to(device)
    print('loaded----')
    filter_bc_policy.load_state_dict(policy.state_dict())
    filter_bc_policy_iw.load_state_dict(policy.state_dict())
    iw_policy.load_state_dict(policy.state_dict())
    old_policy.load_state_dict(policy.state_dict())


    filter_optimizer = torch.optim.Adam(filter_bc_policy.parameters(), lr=filter_lr)
    filter_optimizer_iw = torch.optim.Adam(filter_bc_policy_iw.parameters(), lr=filter_iw_lr)
    filter_sched = WarmupCosineLR(filter_optimizer,
                                  warmup_steps=args.filter_warmup_steps, 
                                  max_steps=args.filter_bc_steps, 
                                  min_lr=0.)
    filter_sched_iw = WarmupCosineLR(filter_optimizer_iw, 
                                     warmup_steps=args.filter_warmup_steps, 
                                     max_steps=args.filter_bc_steps, 
                                     min_lr=0.)
    ema_decay = args.update_q_ema_decay
    print(f'USING EMA iw_policy with decay {ema_decay} normalize={args.normalize}')

    avg_reward = evaluate(device, policy, env, args.seed, 100)
    d4rl_score = env.get_normalized_score(avg_reward) * 100
    print(f'--------BASELINE REWARDS ARE--------{d4rl_score}')
    best_current_score = d4rl_score

    train_iw = args.train_iw
    train_sft = args.train_sft
    evaluate_every = args.filter_log_freq
    for i in range(step_offset, step_offset+args.filter_bc_steps):
        batch = filtered_bc_replay.sample(filter_bs)
        # Use ema update for the importance weighting policy.
        ema_update(iw_policy, filter_bc_policy_iw, ema_decay)
        if train_iw:
            policy_loss_iw, ratio_iw = loss(batch, filter_bc_policy_iw, iw_policy, old_policy, 
                                            temp=args.temp, normalize=args.normalize, 
                                            sum_traj=args.sum_traj)
            lr_iw = filter_optimizer_iw.param_groups[0]['lr']
            filter_optimizer_iw.zero_grad()
            policy_loss_iw.backward()
            filter_optimizer_iw.step()
            filter_sched_iw.step()
            wandb.log(dict(filter_losses_iw=policy_loss_iw, filter_ratios=ratio_iw.mean(),
                        ratio_var=ratio_iw.max() - ratio_iw.min(), learning_rate=lr_iw
                        ), step=i)
        if train_sft:
            policy_loss = loss_bc(batch, filter_bc_policy)
            lr = filter_optimizer.param_groups[0]['lr']
            filter_optimizer.zero_grad()
            policy_loss.backward()
            filter_optimizer.step()
            filter_sched.step()
            wandb.log(dict(filter_bc_losses=policy_loss, learning_rate=lr), step=i)
        
        if i % evaluate_every == 0:
            if train_sft:
                avg_reward = evaluate(device, filter_bc_policy, env, args.seed + i)
                d4rl_score = env.get_normalized_score(avg_reward) * 100
                print(f'----REWARD AT STEP {i}----{d4rl_score}, {policy_loss} lr = {lr}')
                wandb.log(dict(filter_rewards=d4rl_score), step=i)
            if train_iw:
                avg_reward_iw = evaluate(device, filter_bc_policy_iw, env, args.seed + i)
                d4rl_score_iw = env.get_normalized_score(avg_reward_iw) * 100
                wandb.log(dict(filter_rewards_iw=d4rl_score_iw), step=i)
                print(f'----REWARD AT STEP {i}----{d4rl_score_iw}, {policy_loss_iw}, {ratio_iw.mean(), ratio_iw.max(), ratio_iw.min()}') 
    print('Running final evaluation')
    final_step = step_offset + args.filter_bc_steps
    if train_sft:
        avg_reward = evaluate(device, filter_bc_policy, env, final_step, 100)
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        wandb.log(dict(filter_rewards=d4rl_score), step=final_step)
    if train_iw:
        avg_reward_iw = evaluate(device, filter_bc_policy_iw, env, final_step, 100)
        d4rl_score_iw = env.get_normalized_score(avg_reward_iw) * 100
        wandb.log(dict(filter_rewards_iw=d4rl_score_iw), step=final_step)



