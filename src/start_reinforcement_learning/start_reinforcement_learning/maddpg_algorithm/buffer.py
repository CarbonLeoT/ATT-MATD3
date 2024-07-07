import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        # 初始化经验回放缓冲区
        self.mem_size = max_size  # 缓冲区的最大大小
        self.mem_cntr = 0  # 计数器，用于记录存储的经验数量
        self.n_agents = n_agents  # 智能体数量
        self.actor_dims = actor_dims  # 每个智能体的观测空间维度
        self.batch_size = batch_size  # 批量大小
        self.n_actions = n_actions  # 动作空间维度

        # 创建大小为 (mem_size, critic_dims) 的空np数组，用于存储全局状态
        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    # 为每个智能体创建独立的经验回放缓冲区
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    # 将经验存储到np数组中
    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size

        # 将每个智能体的观测存储到各自的缓冲区中
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        # 将全局状态和奖励等存储到中央缓冲区中
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    # 从缓冲区中采样经验
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        # 随机选择一个索引范围，从每个存储的经验中采样
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    # 检查在学习开始之前是否已经存储了足够的样本
    def ready(self):
        return self.mem_cntr >= self.batch_size
