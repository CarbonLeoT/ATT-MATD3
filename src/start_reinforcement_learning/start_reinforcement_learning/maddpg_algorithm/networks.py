import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Critic网络定义
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)

        fc3_dims = 256

        # 定义两个Critic网络的层

        self.fc1 = nn.Linear(input_dims + n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        # 添加注意力层
        #self.attention_3 = AttentionLayer(fc2_dims,fc2_dims)
        #self.layer_4 = nn.Linear(fc3_dims, 1)

        

        """
        self.layer_5 = nn.Linear(input_dims, fc1_dims)
        self.layer_6_s = nn.Linear(fc1_dims, fc2_dims)
        self.layer_6_a = nn.Linear(n_agents * n_actions, fc2_dims)
        # 添加注意力层
        self.attention_7 = AttentionLayer(fc2_dims,fc2_dims)
        self.layer_8 = nn.Linear(fc2_dims, 1)
        #self.layer_8 = nn.Linear(fc3_dims, 1)
        """


        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')

        # 将网络移到指定设备
        self.to(self.device)

    # 前向传播函数，返回状态-动作值Q(s,a)
    def forward(self, s, a):
        x = F.relu(self.fc1(T.cat([s, a], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        """
        s2 = F.relu(self.layer_5(s))
        self.layer_6_s(s2)
        self.layer_6_a(a)
        s21 = T.mm(s2, self.layer_6_s.weight.data.t())
        s22 = T.mm(a, self.layer_6_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_6_a.bias.data)
        s2 = self.attention_7(s2)
        q2 = self.layer_8(s2)
        """

        return q

    # 保存模型检查点
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    # 加载模型检查点
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

# Actor网络定义
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()
        
        fc3_dims = 256
        
        self.chkpt_file = os.path.join(chkpt_dir, name)

        # 定义Actor网络的层
        # 添加注意力层
        # self.attention = AttentionLayer(input_dims,fc3_dims)
        self.layer_1 = nn.Linear(input_dims, fc1_dims)
        self.layer_2 = nn.Linear(fc1_dims, fc2_dims)
        self.layer_3 = nn.Linear(fc2_dims, n_actions)

        self.tanh = nn.Tanh()

        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')

        # 将网络移到指定设备
        self.to(self.device)

    # 前向传播函数，返回动作
    def forward(self, s):
        #s = self.attention(s)
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.layer_3(s)
        a = self.tanh(a)
        return a

    # 保存模型检查点
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    # 加载模型检查点
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

# 引入注意力层
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.sqrt_d = T.sqrt(T.tensor(hidden_dim, dtype=T.float32))

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # Xavier 初始化
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

        # 增大最后两个输入的权重
        with T.no_grad():
            self.query.weight[:, -2:] *= 10
            self.key.weight[:, -2:] *= 10
            self.value.weight[:, -2:] *= 10

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力权重
        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / self.sqrt_d, dim=-1)
        # 计算加权输出
        attention_output = attention_weights @ V

        return attention_output


