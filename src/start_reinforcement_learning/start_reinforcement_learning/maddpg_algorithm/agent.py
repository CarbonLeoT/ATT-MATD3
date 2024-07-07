import torch as T
from start_reinforcement_learning.maddpg_algorithm.networks import ActorNetwork, CriticNetwork  # type: ignore
import numpy as np
from torch.utils.tensorboard import SummaryWriter



class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                 alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.95, tau=0.01, noise_std_dev = 0.5, node_logger=None):
        self.single_robot_ray_space = 36
        self.individual_robot_action_space = 2
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.noise_std_dev = noise_std_dev
        self.expl_min = 0.1
        self.expl_decay_episode_steps = 20000
        self.random_action_prob = 0.5
        self.random_action_prob_min = 0
        self.writer = SummaryWriter(log_dir='data/tensor')
        self.logger = node_logger

        # 定义actor网络和target actor网络
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')

        # 定义critic网络和target critic网络
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                      chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                             chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')

        # 初始化网络参数
        self.update_network_parameters(tau = 1)

    # 根据观察选择动作
    def choose_action(self, observation):
        
        # 更新探索率
        if self.random_action_prob > self.random_action_prob_min:
                self.random_action_prob = self.random_action_prob - ((1 - self.random_action_prob_min) / self.expl_decay_episode_steps)
        if self.noise_std_dev > self.expl_min:
                self.noise_std_dev = self.noise_std_dev - ((1 - self.expl_min) / self.expl_decay_episode_steps)

        # 靠近障碍增大探索
        if min(observation[:(self.single_robot_ray_space)]) < 0.8 and np.random.uniform(0, 1) < self.random_action_prob:
            action = np.random.uniform(-1.0, 1.0, size=self.individual_robot_action_space)

            #self.logger.get_logger().info("choose random action " + str(action))
        else:
            state = T.tensor(observation[np.newaxis, :], dtype=T.float, device=self.actor.device)
            actions = self.actor.forward(state)
            # 生成以选择的动作为均值，方差为self.noise_std_dev的正态分布的最终动作
            std_dev = T.full_like(actions, self.noise_std_dev)
            action = T.normal(actions, std_dev).to(self.actor.device)
            action = T.clamp(action, -1.0, 1.0)
            action = action.detach().cpu().numpy()[0]
            #actions = actions.detach().cpu().numpy()[0]
            #self.logger.get_logger().info("choose action:" + str(actions)+" choose nosie action:" + str(action)+" nosie change:" + str(actions[0]-action[0]))
        return action

    # 更新网络参数
    def update_network_parameters(self, tau=None):
        tau = tau or self.tau

        # 更新actor和target actor的参数
        self._soft_update_network_parameters(self.actor, self.target_actor, tau)

        # 更新critic和target critic的参数
        self._soft_update_network_parameters(self.critic, self.target_critic, tau)


    # 软更新网络参数的辅助函数
    def _soft_update_network_parameters(self, src, dest, tau):
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    # 保存模型
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    # 加载模型
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
