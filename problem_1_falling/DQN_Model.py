import numpy as np
import random
import pickle
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.optim as optim

from DQNAgent.agent_utils import read_cfg, processFrame, statistics, plot_rewardEvaluation
from falling_objects_env import FallingObjects, ACTIONS

from DQNAgent.ReplayBuffer import ReplayBuffer
from DQNAgent.DQN_Network import MyDQN


class MyAgent:
    def __init__(self, nr_actions=2, train=False, use_cuda=False, cfg=None, network_path="QNetwork"):

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
            self.ByteTensor = torch.cuda.ByteTensor
            self.Tensor = self.FloatTensor

        else:
            self.FloatTensor = torch.FloatTensor
            self.LongTensor = torch.LongTensor
            self.ByteTensor = torch.ByteTensor
            self.Tensor = self.FloatTensor

        if train:
            self.cfg = cfg[0]
            self.initial_cfg = cfg[1]
            self.nr_acts = nr_actions
            self.reward_history = []
            self.ep_length = self.cfg.ep_length

        else:
            self.nr_acts = 2
            self.q_network = MyDQN(self.nr_acts)
            with open(network_path, "rb") as f:
                self.q_network.load_state_dict(torch.load(f))

            self.q_network.eval()
            self.q_network.cpu()
            self.replay_memory  = ReplayBuffer(5000, 4)


    def greedyPolicy(self, state, training_model, steps_done):
        '''
        Linear greedy policy
        For the REPLAY_START_SIZE steps explore randomly
        Then explore based on a linear greedy policy from 1 to 0.1 that caps
        after EPS_STABLE steps
        '''
        sample = random.random()
        cfg = self.cfg
        if steps_done < cfg.replay_start_size:
            return 1.0, self.LongTensor([[random.randrange(self.nr_acts)]])

        if steps_done >= cfg.eps_stable:
            eps_threshold = cfg.eps_end
        else:
            eps_decay = -0.9 / cfg.eps_stable
            eps_threshold = cfg.eps_start + steps_done * eps_decay

        if sample > eps_threshold:
            current_state = np.expand_dims(state, axis=0)
            tensor_state = (
                torch.from_numpy(current_state).type(self.Tensor)) / 255.0
            values = training_model(tensor_state).data.max(1)[1].view(1, 1)
            return eps_threshold, values
        else:
            return eps_threshold, self.LongTensor(
                [[random.randrange(self.nr_acts)]])

    def eval_greedyPolicy(self, state, q_network):
        '''
        Linear greedy policy
        For the REPLAY_START_SIZE steps explore randomly
        Then explore based on a linear greedy policy with epsilon 0.05
        for evaluation only
        '''
        sample = random.random()

        if sample > 0.05:
            current_state = np.expand_dims(state, axis=0)
            tensor_state = (
                torch.from_numpy(current_state).type(self.Tensor)) / 255.0
            values = q_network(tensor_state).data.max(1)[1].view(1, 1)
            return values
        else:
            return self.LongTensor([[random.randrange(self.nr_acts)]])

    def optimze_agent(self, q_network, target_q_network, optimizer,
                      replay_memory):

        cfg = self.cfg

        if not replay_memory.can_sample(cfg.batch_size):
            return

        obs_batch, act_batch, rew_batch, nxt_obs_batch, done_mask = replay_memory.sample(
            cfg.batch_size)

        #conver everything to cuda tensors:
        obs_batch = torch.from_numpy(obs_batch).type(self.Tensor) / 255.0
        act_batch = torch.from_numpy(act_batch).unsqueeze(1).type(
            self.LongTensor)
        rew_batch = torch.from_numpy(rew_batch).type(self.Tensor)
        done_mask = torch.from_numpy(done_mask).type(self.ByteTensor)

        nxt_obs_batch = torch.from_numpy(nxt_obs_batch)

        # Compute a mask of non-final states and concatenate the batch elements
        # a byte tensor with true for all non final states
        non_final_next_states = (torch.cat([
            nxt_obs_batch[i].unsqueeze(0) for i in range(done_mask.shape[0])
            if done_mask[i] == 1
        ]).type(self.Tensor)) / 255.0

        # this batch is for computing r + gama * max Q(s_t+1, a: theta_prime)

        # Compute Q(s_t, a; theta) - the model computes Q(s_t), then we select the
        # columns of actions taken given the action batch - more precisely
        # gather only the action value function for the corresponding actions from
        # mini-batch
        state_action_values = q_network(obs_batch).gather(1, act_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(cfg.batch_size).type(self.Tensor)

        next_state_values[done_mask] = target_q_network(
            non_final_next_states).max(1)[0]

        # Compute the expected Q values for target network ( Rt+1 + Gamma * max Q(St+1, A; ThetaPrime))
        expected_state_action_values = (
            next_state_values * cfg.discount_factor) + rew_batch
        expected_state_action_values = expected_state_action_values.detach()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values[:, 0],
                                expected_state_action_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def train(self, env, q_network, optimzer, logger, reloaded=False):

        cfg = self.cfg
        steps_done = 0
        nr_updates = 0
        episode = 0

        target_q_network = MyDQN(self.nr_acts)
        target_q_network.load_state_dict(q_network.state_dict())

        replay_memory = ReplayBuffer(cfg.replay_memory_size,
                                     cfg.agent_history_lenght)

        if reloaded:
            replay_memory.loadMemory()
            with open(cfg.q_network_path, "rb") as f:
                q_network.load_state_dict(torch.load(f))

            with open(cfg.target_network_path, "rb") as f:
                target_q_network.load_state_dict(torch.load(f))

            with open(cfg.extra_params_path, "rb") as f:
                episode, steps_done, nr_updates, self.reward_history = pickle.load(
                    f)

        target_q_network.train(False)
        if self.use_cuda:
            q_network.cuda()
            target_q_network.cuda()

        while episode < cfg.nr_episodes:
            episode += 1
            reward_per_episode = 0
            env = FallingObjects(self.initial_cfg)
            obs = env.reset()
            ep_steps = 0
            while True:
                steps_done += 1
                ep_steps += 1
                #return an 84 * 84 * 1 image
                current_frame = processFrame(obs)
                idx = replay_memory.store_frame(current_frame)

                #get 4 frames stacked together to forward through the network
                current_state = replay_memory.encode_recent_observation()

                eps_threshold, best_action = self.greedyPolicy(
                    current_state, q_network, steps_done)
                obs, rew, done, _ = env.step(best_action[0, 0].item() + 2)

                if rew == 0:
                    reward_per_episode += 1
                    rew = 1
                else:
                    done = True

                replay_memory.store_effect(idx, best_action[0, 0], rew, done)

                #Let the agent explore randomly for REPLAY_START_SIZE steps
                if steps_done > cfg.replay_start_size:

                    if steps_done % 4 == 0:

                        nr_updates += 1
                        self.optimze_agent(q_network, target_q_network,
                                           optimzer, replay_memory)

                        if nr_updates % cfg.update_target_network == 0:
                            logger.write("Updated target network " +
                                         str(episode) + "\n")
                            print("Updated target network" + str(episode))
                            logger.flush()
                            target_q_network.load_state_dict(
                                q_network.state_dict())

                        plot_rewardEvaluation(cfg, self, nr_updates, q_network, logger)

                if done:
                    break

            if episode > 0 and episode % 50 == 0:
                #save replay_memory
                replay_memory.saveMemory()

                #save networks
                with open(cfg.q_network_path, "wb") as f:
                    torch.save(q_network.state_dict(), f)

                with open(cfg.target_network_path, "wb") as f:
                    torch.save(target_q_network.state_dict(), f)

                with open(cfg.extra_params_path, "wb") as f:
                    extra_params = [
                        episode, steps_done, nr_updates, self.reward_history
                    ]
                    pickle.dump(extra_params, f)
                logger.write("Saved networks and parameters after " +
                             str(episode) + "\n")
                logger.flush()

            statistics(episode, steps_done, nr_updates, reward_per_episode, eps_threshold, ep_steps)

    def evalAgent(self, nr_episodes, q_network):

        replay_memory = ReplayBuffer(5000, 4)
        episode = 0
        episodes_score = []

        env = FallingObjects(self.initial_cfg)

        while episode < nr_episodes:
            print(episode)
            episode += 1
            reward_per_episode = 0
            env = FallingObjects(self.initial_cfg)
            obs = env.reset()
            while True:

                #return  an 84 * 84 *1 image
                current_frame = processFrame(obs)
                idx = replay_memory.store_frame(current_frame)

                #get 4 frames stacked together to forward throgh network
                current_state = replay_memory.encode_recent_observation()
                best_action = self.eval_greedyPolicy(current_state, q_network)

                obs, rew, done, _ = env.step(best_action[0, 0].item() + 2)
                if rew == 0:
                    reward_per_episode += 1
                    rew  = 1
                    replay_memory.store_effect(idx, best_action[0, 0], rew, done)
                else:
                    done = True
                    replay_memory.store_effect(idx, best_action[0, 0], rew, done)
                    break


            episodes_score.append(reward_per_episode)

        mean_score = sum(episodes_score) / float(nr_episodes)
        return mean_score, episodes_score

    def act(self, observation):

        #return  an 84 * 84 *1 image
        current_frame = processFrame(observation)
        idx = self.replay_memory.store_frame(current_frame)

        #get 4 frames stacked together to forward throgh network
        current_state = self.replay_memory.encode_recent_observation()
        best_action = self.eval_greedyPolicy(current_state, self.q_network)

        self.replay_memory.store_effect(idx, best_action[0, 0], 1, False)

        return best_action[0, 0].item() + 2


def main_function():

    arg_parser = ArgumentParser(description="Train DQN network")
    arg_parser.add_argument(
        '-c',
        '--config-file',
        default='configs/default.yaml',
        type=str,
        dest='config_file',
        help='Default configuration file')

    arg_parser.add_argument(
        "--agent_file",
        default='DQNAgent/agent_config.yaml',
        type=str,
        help='Agent configuration file')

    arg_parser.add_argument("--optimizer", type=str, default="Adam")
    arg_parser.add_argument("--new", default=False, action='store_true')
    arg_parser.add_argument("--use_cuda", default=False, action='store_true')

    args = arg_parser.parse_args()

    config_file = args.config_file
    cfg = read_cfg(config_file)

    acfg = read_cfg(args.agent_file)

    env = FallingObjects(cfg)

    nr_actions = 2
    q_network = MyDQN(nr_actions)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(q_network.parameters(), lr=acfg.adam_lr)

    elif args.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(
            q_network.parameters(),
            lr=acfg.rms_prop_lr,
            alpha=0.99,
            eps=0.01,
            weight_decay=0,
            momentum=acfg.gradient_momentum,
            centered=False)

    agent = MyAgent(nr_actions, True, args.use_cuda, (acfg, cfg))
    with open("log_info", "wt") as f:
        agent.train(env, q_network, optimizer, f, args.new)


if __name__ == '__main__':
    main_function()
