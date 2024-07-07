import torch as T
import torch.nn.functional as F
from start_reinforcement_learning.maddpg_algorithm.agent import Agent # type: ignore
import numpy as np
import torch
from numpy import inf
import time
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

# MATD3算法

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='robot', alpha=0.01, beta=0.01, fc1=512,
                 fc2=512, gamma=0.7, tau=0.01, chkpt_dir='tmp/maddpg/', node_logger=None):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.logger = node_logger
        self.started_learning = False
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.writer = SummaryWriter(log_dir='data/tensor')
        self.Q = 0
        self.loss = 0
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2, gamma=gamma, tau=tau,node_logger=self.logger))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    # 将连续动作离散化
    def discretize(self, continuous_actions):

        # 使用tanh激活函数，continuous_actions 范围为 -1 到 1
        if continuous_actions[0] < -0.33:
            linear_velocity_action = 0
        elif continuous_actions[0] < 0.33:
            linear_velocity_action = 1
        else:
            linear_velocity_action = 2

        angular_velocity_action = continuous_actions[1]
        discrete_actions = np.array([linear_velocity_action, angular_velocity_action])
        return discrete_actions

    # 返回每个agent选择的线速度和角速度动作
    def choose_action(self, raw_obs):
        actions = {}
        for agent_id, agent in zip(raw_obs, self.agents):
            continuous_actions = agent.choose_action(raw_obs[agent_id])
            # self.logger.get_logger().info(str(continuous_actions))
            discrete_actions = self.discretize(continuous_actions)
            actions[agent_id] = discrete_actions
        return actions

    # 调整actor和critic的权重
    def learn(self, memory, learn_step_counter, update_actor_interval):
        # 记录数据
        start_time = time.time()
        # 如果memory没有达到批量大小，则返回
        if not memory.ready():
            return 0

        # 从central memory中采样
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        # 确保每个tensor都在同一个设备上，应该是gpu（cuda:0）
        device = self.agents[0].actor.device

        # 将采样的memory数组列表转换为Tensors
        states = T.tensor(np.array(states), dtype=T.float).to(device)
        rewards = T.tensor(np.array(rewards), dtype=T.float).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float).to(device)
        states_ = T.tensor(np.array(states_), dtype=T.float).to(device)
        dones = T.tensor(np.array(dones)).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # 对每个agent的个体memory
        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states).to(device)

            #noise = torch.Tensor(self.n_actions).data.normal_(0, self.policy_noise).to(device)
            #noise = noise.clamp(-self.noise_clip,self.noise_clip)
            #new_pi = ( new_pi + noise).clamp(-1,1)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states).to(device)
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(actions[agent_idx])

        # 创建一个tensor并存储从每个agent的'next state'中连接的动作概率
        new_actions = T.cat([a for a in all_agents_new_actions], dim=1)
        # 同上，但针对'state'
        mu = T.cat([a for a in all_agents_new_mu_actions], dim=1)
        # 创建一个tensor并存储从'previous state'中每个agent执行的动作
        old_actions = T.cat([a for a in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # 计算critic-target网络的Q值
            # target_Q1, target_Q2 = agent.target_critic.forward(states_, new_actions)
            target_Q = agent.target_critic.forward(states_, new_actions).squeeze()


            # 从两个Q值中选择最小的
            # target_Q = T.min(target_Q1, target_Q2).squeeze()
            
            # 计算当前参数下基础网络的Q值
            # current_Q1, current_Q2 = agent.critic(states, old_actions)
            current_Q = agent.critic.forward(states, old_actions).squeeze()        
            
            # 更新已结束的状态对应的动作值为零
            target_Q[dones[:, 0]] = 0.0

            # 计算目标网络的最终Q值
            target = rewards[:, agent_idx] + agent.gamma * target_Q.detach()


            # 计算当前Q值和目标Q值之间的损失
            #critic_loss = (F.mse_loss(current_Q1.squeeze(), target) + F.mse_loss(current_Q2.squeeze(), target))  * 0.5
            critic_loss = F.mse_loss(current_Q, target)

            # 计算演员网络的损失
            # actor_loss, _ = agent.critic.forward(states, mu)
            actor_loss = -agent.critic.forward(states, mu).flatten()
            actor_loss = T.mean(actor_loss)

            # 清除之前的梯度
            agent.critic.optimizer.zero_grad()
            
            # 计算critic网络的梯度
            critic_loss.backward(retain_graph=True)
            #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)


            if learn_step_counter % update_actor_interval == 0:
                # 清除之前的梯度
                agent.actor.optimizer.zero_grad()
                # 计算actor网络的梯度
                actor_loss.backward(retain_graph=True)
            

        # 应用梯度并更新网络参数
        for agent_idx, agent in enumerate(self.agents):
            if learn_step_counter % update_actor_interval == 0:
                agent.actor.optimizer.step()
            agent.critic.optimizer.step()
            # 更新目标网络的参数
            agent.update_network_parameters()

        # 记录数据
        self.writer.add_scalar("critic loss", critic_loss , learn_step_counter)
        self.writer.add_scalar("actor loss", actor_loss , learn_step_counter)
        self.writer.add_scalar("Q", torch.mean(target_Q) ,learn_step_counter)
        end_time = time.time()
        # 计算学习所需时间
        learn_time = end_time - start_time
        self.writer.add_scalar("learn time", learn_time ,learn_step_counter)
        #msg = 'Time taken for learn step: '+str(round(learn_time,4))
        #self.logger.get_logger().info(msg)
