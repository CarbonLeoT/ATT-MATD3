import os
import rclpy
from rclpy.node import Node
import numpy as np
import time
from std_srvs.srv import Empty
import rclpy.service
from start_reinforcement_learning.env_logic.logic import Env  # type: ignore
from start_reinforcement_learning.maddpg_algorithm.maddpg import MADDPG  # type: ignore
from start_reinforcement_learning.maddpg_algorithm.buffer import MultiAgentReplayBuffer  # type: ignore
import torch as T
import gc
from ament_index_python.packages import get_package_share_directory
from torch.utils.tensorboard import SummaryWriter


# 将列表的观察值数组转换为一个平铺的状态向量
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# 运行MADDPG算法的主要函数
class MADDPGNode(Node):
    def __init__(self, map_number, robot_number, gazebo_control):
        super().__init__('maddpg_node')

        self.gazebo_control = gazebo_control
        self.writer = SummaryWriter(log_dir='data/tensor')
        # 从launch文件中获取传递的参数
        map_number = self.declare_parameter('map_number', 1).get_parameter_value().integer_value
        robot_number = self.declare_parameter('robot_number', 3).get_parameter_value().integer_value

        self.get_logger().info(f"Map number: {map_number}")
        self.get_logger().info(f"Robot number: {robot_number}")

        #T.cuda.empty_cache()
        #gc.collect()

        # 使用动作大小设置环境
        env = Env(robot_number, map_number)
        self.get_logger().info(f"Map number: {map_number}")
        n_agents = env.number_of_robots

        actor_dims = env.observation_space()
        critic_dims = sum(actor_dims)

        # 设置动作空间
        n_actions = env.action_space()

        chkpt_dir_var = os.path.join(get_package_share_directory('start_reinforcement_learning'),
                                    'start_reinforcement_learning','deep_learning_weights','maddpg')

        # 初始化主要算法
        maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                               fc1=800, fc2=600, gamma=0.8, tau=0.005 ,
                               alpha=1e-3, beta=1e-2, scenario='robot',
                               chkpt_dir=chkpt_dir_var, node_logger=self)

        # 初始化内存
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                            n_actions, n_agents, batch_size = 2000)

        PRINT_INTERVAL = 10
        N_GAMES = 3000
        total_steps = 0
        score_history = []
        evaluate = False
        best_score = 0
        learn_step_counter = 0
        update_actor_interval = 1
        TIME_DELTA = 0.3

        # 测试网络，训练包括critic和actor，测试仅包括actor
        if evaluate:
            maddpg_agents.load_checkpoint()

        # 训练过程
        for i in range(N_GAMES):
            self.gazebo_control.unpause_gazebo()
            # 重置以获取初始观察
            obs = env.reset()
            # 将字典转换为数组列表以进入'obs_list_to_state_vector'函数
            list_obs = list(obs.values())
            score = 0
            done = [False] * n_agents
            terminal = [False] * n_agents
            episode_step = 0
            run_ep = True
            # truncated表示集已达到最大步数，done表示碰撞或到达目标
            while not any(terminal):
                time_s = time.time() 
                # 获取算法认为在给定观察中最好的动作
                actions = maddpg_agents.choose_action(obs)
                
                # 使用step函数获取下一个状态和奖励信息以及集是否“done”
                obs_, reward, done, truncated, info = env.step(actions)
                #self.get_logger().info('reward = ' + str(reward))
                # 将字典转换为数组列表以进入'obs_list_to_state_vector'函数
                list_done = list(done.values())
                list_reward = list(reward.values())
                list_actions = list(actions.values())
                list_obs_ = list(obs_.values())
                list_trunc = list(truncated.values())
                # 将数组列表转换为一个平铺的状态向量
                state = obs_list_to_state_vector(list_obs)
                state_ = obs_list_to_state_vector(list_obs_)

                terminal = [d or t for d, t in zip(list_done, list_trunc)]
                # 将原始观察以及每个代理的观察、奖励和完成值列表一起存储
                memory.store_transition(list_obs, state, list_actions, list_reward, list_obs_, state_, list_done)

                if total_steps % 100 == 0 and not evaluate:
                    self.gazebo_control.pause_gazebo()
                    if maddpg_agents.learn(memory, learn_step_counter, update_actor_interval) != 0 :
                        learn_step_counter += 1
                    self.gazebo_control.unpause_gazebo()


                # 设置新的观察为当前观察
                obs = obs_
                score += sum(list_reward)
                total_steps += 1
                episode_step += 1
                
                
                # 计算每步时长
                time.sleep(TIME_DELTA)
                time_e = time.time()

                #self.get_logger().info('step time = ' + str(time_e - time_s))


            # 计算每个机器人的平均得分
            self.writer.add_scalar("reward", score / robot_number ,i)
            score_history.append(score / robot_number)
            # 平均最近100次的得分
            avg_score = np.mean(score_history[-10:])

            if i % PRINT_INTERVAL == 0 and i > 0:
                self.get_logger().info(
                    'Episode: {}, Average score: {:.1f}, learn steps: {}'.format(i, avg_score, learn_step_counter))
            


def main(args=None):
    rclpy.init(args=args)
    map_number = int(os.getenv('map_number', '1'))
    robot_number = int(os.getenv('robot_number', '3'))
    gazebo_control = GazeboControl()
    node = MADDPGNode(map_number, robot_number ,gazebo_control)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# 控制gazebo仿真的类
class GazeboControl(Node):
    def __init__(self):
        super().__init__('gazebo_control')
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

    # 等待服务可用
    def call_service(self, client):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        try:
            req = Empty.Request()
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is  None:
                self.get_logger().error('Service call failed')
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    # 暂停和继续Gazebo仿真的函数
    def pause_gazebo(self):
        self.call_service(self.pause_client)
        #self.get_logger().info('pause')
    def unpause_gazebo(self):
        self.call_service(self.unpause_client)
        #self.get_logger().info('unpause')

if __name__ == '__main__':
    main()
