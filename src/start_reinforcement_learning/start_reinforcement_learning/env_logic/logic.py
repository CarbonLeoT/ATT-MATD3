import rclpy
from rclpy.node import Node
from PIL import Image,ImageDraw
import numpy as np
import math
from math import pi
import time
import os
import pandas as pd
from squaternion import Quaternion # type: ignore

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.qos import qos_profile_sensor_data

from start_reinforcement_learning.env_logic.restart_environment import RestartEnvironment  # type: ignore


class Env():
    def __init__(self, number_of_robots=3, map_number=1):
        self.number_of_robots = number_of_robots
        self.map_number = map_number
        self.restart_environment = RestartEnvironment(self.number_of_robots, self.map_number)
        # 创建一个列表，用于存储发布速度指令的节点
        self.cmd_vel_publisher_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.cmd_vel_publisher_list[i] = PublishCMD_VEL(i)

        # 创建一个列表，用于读取每个机器人的里程计数据
        self.odometry_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.odometry_subscriber_list[i] = ReadOdom(i)

        # 创建一个列表，用于读取激光扫描信息
        self.scan_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.scan_subscriber_list[i] = ReadScan(i)

        # 创建一个用于记录日志的节点
        self.logger = Logger()

        # 36位激光雷达 + 自己的速度信息 + 目标距离 + 目标朝向 = 36+2+2
        self.single_robot_ray_space = 36
        self.individual_robot_action_space = 2
        self.additional_observation_space_per_robot = 2
        self.single_robot_observation_space = self.single_robot_ray_space + self.individual_robot_action_space + self.additional_observation_space_per_robot
        self.total_robot_observation_space = []

        for _ in range(self.number_of_robots):
            self.total_robot_observation_space.append(self.single_robot_observation_space)
        self.initGoal = True
        self.current_goal_location = []
        self.current_scan_data = 0

        self.step_counter = 0
        self.reached_goal_counter = 0
        self.total_goal_counter = 0
        self.MAX_STEPS = 500

        # 机器人属性列表
        self.current_angular_velocity = np.zeros(self.number_of_robots)
        self.current_linear_velocity = np.zeros(self.number_of_robots)
        self.current_pose_x = np.zeros(self.number_of_robots)
        self.current_pose_y = np.zeros(self.number_of_robots)
        self.current_orientation_x = np.zeros(self.number_of_robots)
        self.current_orientation_y = np.zeros(self.number_of_robots)
        self.current_orientation_z = np.zeros(self.number_of_robots)
        self.current_orientation_w = np.zeros(self.number_of_robots)
        self.current_observations = np.zeros(self.number_of_robots)
        self.distance = np.zeros(self.number_of_robots)
        

        # 机器人速度约束
        self.max_linear_vel = 0.60
        self.min_linear_vel = 0.15
        self.max_angular_vel = 0.25
        self.min_angular_vel = -0.25

        # 轨迹列表
        self.trajectories = {i: [] for i in range(self.number_of_robots)}

        # 创建一个文件夹用于保存评价参数
        current_directory = os.path.dirname(__file__)
        self.folder_data_path = 'data'
        if not os.path.exists(self.folder_data_path):
            os.makedirs(self.folder_data_path)
        self.excel_path = os.path.join(self.folder_data_path, "simulation_records.xlsx")
        self.episode = -1
        self.rewards = 0
        self.coverage = 0
        self.collisions = 0
        self.steps = 0
        self.goals = 0
        self.coverageRewards = 0
        self.velocityRewards = 0
        self.coveragePenaltys = 0
        self.distanceRewards = 0

        # 奖励系数
        self.goalReward = 100
        self.collisionReward = -100
        self.coverageReward = 0
        self.velocityReward = 0
        self.coveragePenalty = 0
        self.distanceReward = 20

        # 期望速度
        self.target_linear_velocity = 0.2
        self.target_angular_velocity = 0

        # 地图数据
        self.mapfactor = 100
        self.map_x = 10 * self.mapfactor
        self.map_y = 10 * self.mapfactor
        self.map_size = self.map_x * self.map_y
        self.covermap = np.zeros([self.map_x, self.map_y])

        # 图片顺序
        self.imgorder = 0

        # 覆盖参数
        self.coveragerad = 1
        self.coverage_1 = 0
        self.coverage_2 = 0
        self.Repeated_times = 50

        # 创建一个文件夹用于保存图片
        current_directory = os.path.dirname(__file__)
        self.folder_image_path = 'image'
        if not os.path.exists(self.folder_image_path):
            os.makedirs(self.folder_image_path)

    # 转换坐标系    
    def x_tran(self,x):
        return int((x + 5) * self.mapfactor)
    def y_tran(self,y):
        return int((y)* -1 * self.mapfactor)
    
    # 获取观测空间。
    def observation_space(self):
        return self.total_robot_observation_space

    def action_space(self):
        return self.individual_robot_action_space

    # 获取目标与最近机器人的距离（调试函数）
    def getGoalDistace(self):
        closest_distance_to_goal = math.inf
        for i in range(self.number_of_robots):
            ith_robots_distance = round(math.hypot(
                self.current_goal_location[0] - self.current_pose_x[i],
                self.current_goal_location[1] - self.current_pose_y[i]), 2)
            if ith_robots_distance < closest_distance_to_goal:
                closest_distance_to_goal = ith_robots_distance
        return closest_distance_to_goal

    # 处理雷达数据
    def resize_lidar(self, scan):
        scan_range = []
        ray_scan = np.empty(360 ,dtype=np.float32)
 
        # 调整扫描范围数据的大小，scan本身返回有关扫描的额外信息，scan.ranges只获取范围数据
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
                ray_scan[i] = 3.5
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
                ray_scan[i] = 0
                print('broken')
            else:
                scan_range.append(scan.ranges[i])
                ray_scan[i] = scan.ranges[i]

        # 将360度数据降维到36个维度
        np_scan_range = np.empty(self.single_robot_observation_space, dtype=np.float32)

        for i in range(0, 36):
            j = 10*i
            min_value = min(ray_scan[j:j+10])
            np_scan_range[i] = min_value

        return np_scan_range

    # 重置所有机器人的速度，在环境重置后调用此函数
    def reset_cmd_vel(self):
        for i in range(self.number_of_robots):
            # 重置cmd_vel并发布
            self.current_linear_velocity[i] = 0
            self.current_angular_velocity[i] = 0
            cmd_vel_publisher = self.cmd_vel_publisher_list[i]
            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(0)
            desired_vel_cmd.angular.z = float(0)
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()
        time.sleep(1)

    # 检查给定的扫描是否显示碰撞
    def hasCollided(self, scan_range,robot_number):
        min_range = 0.35
        # 仅检查扫描结果，不包括速度
        if min_range > np.min(scan_range[:(self.single_robot_ray_space)]) > 0.0:
            self.collisions += 1
            return True
        #msg = 'scan_all =' + str(scan_range)
        #self.logger.log(msg)
        return False

    # 检查当前机器人的位置是否到达目标
    def hasReachedGoal(self, scan_range, robot_number):
        dis_to_goal = round(math.hypot(
            self.current_goal_location[0] - self.current_pose_x[robot_number],
            self.current_goal_location[1] - self.current_pose_y[robot_number]), 2)
        if dis_to_goal < self.coveragerad:
            self.goals += 1
            return True
        return False

    # 计算覆盖率
    def calculate_coverage(self):
        total_cells = self.covermap.shape[0] * self.covermap.shape[1]  # 计算总单元格数
        covered_cells = 0  # 初始化被覆盖的单元格计数

        # 遍历每个单元格
        for x in range(self.covermap.shape[0]):
            for y in range(self.covermap.shape[1]):
                if self.covermap[x, y] >= 1:
                    covered_cells += 1  # 如果单元格被覆盖至少一次，增加计数

        coverage_rate = covered_cells / total_cells  # 计算覆盖率
        return coverage_rate

    # 定义惩罚函数
    def calculate_penalty(self, count):
        return count
      
    # 加入离目标的距离作为奖励
    def getRewards(self,scan_range):
        robotRewards = []
        startingReward = 0 # 初始奖励
        for i in range(self.number_of_robots):
            
            min_laser = np.min(scan_range[i][:(self.single_robot_ray_space)])
            currentRewards = startingReward

            r3 = lambda x: 1 - x if x < 1 else 0.0
            currentRewards += self.velocityReward * (self.current_linear_velocity[i]  -
                        abs(self.current_angular_velocity[i]) - r3(min_laser))/2
            
            self.velocityRewards += currentRewards

            currentRewards_ = self. distanceReward * (self.distance[i] - self.calculate_goal_distance(i))

            self.distanceRewards += currentRewards_

            currentRewards = round(currentRewards + currentRewards_, 5)

            # 清除异常值
            if currentRewards >= 1 or currentRewards <= -1:
                currentRewards = 0    
            
            robotRewards.append(currentRewards)
            self.distance[i] = self.calculate_goal_distance(i)
        #self.logger.log(str(robotRewards))
        return robotRewards

    # 将数组列表转换为字典
    def handleReturnValues(self, robotScans, robotRewards, robotDones, truncated, info):
        # 每个机器人的观测字典
        robot_observations = {}
        robot_rewards = {}
        robot_dones = {}
        robot_truncated = {}
        info = {}
        for i, val in enumerate(robotScans):
            robot_observations['robot' + str(i)] = val
        for i, val in enumerate(robotRewards):
            robot_rewards['robot' + str(i)] = val
        for i, val in enumerate(robotDones):
            robot_dones['robot' + str(i)] = val
        for i, val in enumerate(truncated):
            robot_truncated['robot' + str(i)] = val
        # msg = str(robot_rewards)
        # self.logger.log(msg)
        return robot_observations, robot_rewards, robot_dones, robot_truncated, info

    # 更新包含机器人位置和朝向的变量
    def updateRobotPosition(self):
        for i in range(self.number_of_robots):
            odom_data = None
            odom_subscriber = self.odometry_subscriber_list[i]
            while odom_data is None:
                rclpy.spin_once(odom_subscriber)
                odom_data = odom_subscriber.odom
            self.current_pose_x[i] = odom_data.pose.pose.position.x
            self.current_pose_y[i] = odom_data.pose.pose.position.y
            self.current_orientation_x[i] = odom_data.pose.pose.orientation.x
            self.current_orientation_y[i] = odom_data.pose.pose.orientation.y
            self.current_orientation_z[i] = odom_data.pose.pose.orientation.z
            self.current_orientation_w[i] = odom_data.pose.pose.orientation.w




    # 获取目标与机器人的距离
    def  calculate_goal_distance(self, robot_index):
        distance = round(math.hypot(
            self.current_goal_location[0] - self.current_pose_x[robot_index],
            self.current_goal_location[1] - self.current_pose_y[robot_index]), 5)
        return distance

    # 获取目标与机器人的朝向
    def calculate_goal_direction(self, robot_index):
        quaternion = Quaternion(
            self.current_orientation_w[robot_index],
            self.current_orientation_x[robot_index],
            self.current_orientation_y[robot_index],
            self.current_orientation_z[robot_index],
            )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        skew_x = self.current_goal_location[0]- self.current_pose_x[robot_index]
        skew_y = self.current_goal_location[1] - self.current_pose_y[robot_index]
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = theta - 2 * np.pi
        if theta < -np.pi:
            theta = theta + 2 * np.pi
        return theta

    # 添加信息到观测空间
    def addVelocitiesToObs(self, scans):
        for i in range(self.number_of_robots):
            index = self.single_robot_ray_space
            # 添加自身速度信息
            scans[i][index] = round(self.current_linear_velocity[i], 4)
            scans[i][index + 1] = round(self.current_angular_velocity[i], 4)
            # 添加目标信息
            scans[i][index + 2] = round(self.calculate_goal_distance(i), 4)
            scans[i][index + 3] = round(self.calculate_goal_direction(i), 4)
        return scans

    def end_of_episode_functions(self, robot_scans):
        # 快速更新机器人的位置变量，然后重置速度
        self.updateRobotPosition()
        self.reset_cmd_vel()
        # 将速度添加到观测空间的末尾
        self.addVelocitiesToObs(robot_scans)

    # 写入excel数据
    def append_to_excel(self, df):
        # 检查文件是否存在
        if not os.path.exists(self.excel_path):
            # 文件不存在，创建新文件并写入数据
            with pd.ExcelWriter(self.excel_path, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Simulation Data')
        else:
            # 文件存在，追加数据
            with pd.ExcelWriter(self.excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                workbook = writer.book
                if 'Simulation Data' in workbook.sheetnames:
                    # 获取工作表对象
                    worksheet = workbook['Simulation Data']
                    # 计算该工作表的最大行数，即新数据应开始写入的行
                    startrow = worksheet.max_row
                    write_header = False if startrow > 0 else True
                else:
                    startrow = 0  # 如果工作表不存在，则从第一行开始写
                    write_header = True

                # 追加数据到指定工作表，从计算出的行数开始
                df.to_excel(writer, index=False, sheet_name='Simulation Data', startrow=startrow, header=write_header)

    # 步进环境
    def step(self, action):
        self.updateRobotPosition()

        # 进行覆盖
        for i in range(self.number_of_robots):
            x_transform = self.x_tran(self.current_pose_x[i])
            y_transform = self.y_tran(self.current_pose_y[i])
            radius = int(0.4 * self.mapfactor)  # 圆的半径，乘以转换到地图的尺度
            radius_squared = radius ** 2  # 预计算半径的平方，避免开方操作

            # 遍历圆形覆盖区域
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx ** 2 + dy ** 2 <= radius_squared:  # 使用平方比较
                        covermap_x = x_transform + dx
                        covermap_y = y_transform + dy
                        # 检查坐标是否在地图范围内
                        if 0 <= covermap_x < self.map_x and 0 <= covermap_y < self.map_y:
                            self.covermap[covermap_y, covermap_x] = 1
        
        # 记录轨迹
        for i in range(self.number_of_robots):
            x = self.current_pose_x[i]
            y = self.current_pose_y[i]
            if -5 <= x <= 5 and -10 <= y <= 0:
                self.trajectories[i].append((x, y))


        truncated = [False] * self.number_of_robots
        dones = [False] * self.number_of_robots
        info = {}
        # 读取所有机器人的激光扫描数据
        robot_scans = []
        for i in range(self.number_of_robots):
            data = None
            scan_data = self.scan_subscriber_list[i]

            while data is None:
                rclpy.spin_once(scan_data)
                data = scan_data.scan

            scan_range = self.resize_lidar(data)
            robot_scans.append(scan_range)
        # 如果达到最大步数，则返回truncated为True
        if self.step_counter + 1 > self.MAX_STEPS:
            truncated = [True] * self.number_of_robots
            rewards = self.getRewards(robot_scans)
            self.end_of_episode_functions(robot_scans)
            # 返回数据
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
        # 检查是否有机器人达到了目标或发生了碰撞
        collided = np.full(self.number_of_robots, False)
        reachedGoal = np.full(self.number_of_robots, False)
        for i in range(self.number_of_robots):
            collided[i] = self.hasCollided(robot_scans[i], i) 
            reachedGoal[i] = self.hasReachedGoal(robot_scans[i], i)
        # 检查是机器人都到达目标
        if any(reachedGoal):
            # self.logger.log('A robot has found the goal')
            self.reached_goal_counter += 1
            self.total_goal_counter += 1
            # 如果是这样，快速获取基本奖励
            rewards = self.getRewards(robot_scans)
            # 获取到达目标的机器人的索引
            indexes = np.nonzero(reachedGoal)[0]
            for idx in indexes:
                rewards[idx] += self.goalReward
                # 将完成值设置为True，表示机器人已到达目标
                dones[idx] = True
            self.rewards += sum(rewards) / self.number_of_robots
            self.end_of_episode_functions(robot_scans)
            # 返回数据
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)
        # 如果没有机器人到达目标，则检查是否机器人都发生了碰撞

        # 有时机器人加速过快，它们的框架倾斜，导致传感器射线击中地面
        # 可以调整机器人的惯性属性解决
        if any(collided):
            # 获取基本奖励
            rewards = self.getRewards(robot_scans)
            # 获取发生碰撞的机器人的索引
            indexes = np.nonzero(collided)[0]
            for idx in indexes:
                rewards[idx] += self.collisionReward
                # 将完成值设置为True，表示机器人发生碰撞
                dones[idx] = True
            self.end_of_episode_functions(robot_scans)
            self.rewards += sum(rewards) / self.number_of_robots
            # 返回数据
            return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)

        # 如果没有机器人到达目标或发生碰撞，正常继续

        # 对于每个机器人，获取其选择的动作并传递给gazebo
        # TODO--动作设计有问题，存在加速过快导致机器人倾斜的问题，仿真中可以运行，实物可能存在问题

        for i in range(self.number_of_robots):
            name = 'robot' + str(i)

            # 动作
            chosen_linear_action = action[name][0]
            chosen_angular_action = action[name][1]

            
            # chosen_angular_action为-1到1的量
            if chosen_linear_action == 2:
                self.current_linear_velocity[i] += 0.02
            if chosen_linear_action == 1:
                self.current_linear_velocity[i] += 0.0
            if chosen_linear_action == 0:
                self.current_linear_velocity[i] -= 0.02
                
            self.current_angular_velocity[i] = chosen_angular_action

            cmd_vel_publisher = self.cmd_vel_publisher_list[i]

            self.current_linear_velocity[i] = max(
                min(self.current_linear_velocity[i], self.max_linear_vel), self.min_linear_vel)
            self.current_angular_velocity[i] = max(
                min(self.current_angular_velocity[i], self.max_angular_vel), self.min_angular_vel)

            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(self.current_linear_velocity[i])
            desired_vel_cmd.angular.z = float(self.current_angular_velocity[i])
            # 发布动作
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()
        rewards = self.getRewards(robot_scans)
        dones = [False] * self.number_of_robots
        self.updateRobotPosition()
        
        # 将速度添加到观测空间的末尾
        self.addVelocitiesToObs(robot_scans)
        self.step_counter += 1
        self.rewards += np.sum(rewards) / self.number_of_robots
        return self.handleReturnValues(robot_scans, rewards, dones, truncated, info)

    # 重置环境
    def reset(self):
        self.episode += 1
        self.coverage = self.coverage_2
        self.steps = self.step_counter

        # 记录新的回合数据
        new_record = {
            'Episode': self.episode,
            'Rewards': round(self.rewards, 4),
            'Collisions': self.collisions,
            'Steps': self.steps,
            'Goals Reached': self.goals,
            'velocityRewards': round(self.velocityRewards, 4),
            'distanceRewards':round(self.distanceRewards, 4)
        }
        new_df = pd.DataFrame([new_record])
        self.append_to_excel(new_df)

        # 生成轨迹图像
        height = self.map_y
        width = self.map_x
        # 创建一个新图像，模式为'RGB'
        img = Image.new('RGB', (width, height), "white")  # 背景设为白色
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                if self.covermap[y][x] == 1:
                    pixels[x, y] = (122, 203, 196)  # 设置已覆盖的区域
                else:
                    pixels[x, y] = (246, 251, 239)  # 设置未覆盖的区域

        draw = ImageDraw.Draw(img) 

        # 绘制障碍物位置
        if self.episode >= 1:
            cylinker_x = [-2,0,3]
            cylinker_y = [-2,-5,-7]
            for i in range (len(cylinker_x)):
                cylinker_x[i] = self.x_tran(cylinker_x[i])
                cylinker_y[i] = self.y_tran(cylinker_y[i])
                r = 0.5 * self.mapfactor
                draw.ellipse((cylinker_x[i] - r, cylinker_y[i] - r, cylinker_x[i]  + r, cylinker_y[i] + r), outline='black', width=2000)     
        
        # 绘制目标位置
        if self.episode >= 1:
            goal_x = self.current_goal_location[0]
            goal_y = self.current_goal_location[1]
            goal_x = self.x_tran(goal_x)
            goal_y = self.y_tran(goal_y)
            r = self.coveragerad * self.mapfactor
            draw.ellipse((goal_x - r, goal_y - r, goal_x  + r, goal_y + r), outline='orange', width=2000)       

        # 在图像上绘制轨迹
        colors = ['red', 'green', 'blue']
        for i in range(self.number_of_robots):
            trajectory = self.trajectories[i]
            for j in range(0, (len(trajectory) - 1)):
                x1, y1 = trajectory[j]
                x2, y2 = trajectory[j + 1]

                x1 = self.x_tran(x1)
                x2 = self.x_tran(x2)
                y1= self.y_tran(y1)
                y2= self.y_tran(y2)
                
                draw.line((x1, y1, x2, y2), fill=colors[i], width=2)
            if trajectory:
                # 标注起始位置
                x_start, y_start = trajectory[0]
                x_start = self.x_tran(x_start)
                y_start = self.y_tran(y_start)

                draw.ellipse((x_start - 15, y_start - 15, x_start + 15, y_start + 15), outline=colors[i], width=100)

        
        # img.show()  # 显示图像

        # 设置图像保存路径
        save_path = os.path.join(self.folder_image_path, "image" + str(self.imgorder) + ".png")
        img.save(save_path)
        self.imgorder += 1

        # 重置各种统计数据
        self.rewards = 0
        self.coverage = 0
        self.collisions = 0
        self.steps = 0
        self.goals = 0
        self.coverageRewards = 0
        self.velocityRewards = 0
        self.coveragePenaltys = 0
        self.distanceRewards = 0
        self.coverage_1 = 0
        self.coverage_2 = 0
        self.covermap = np.zeros([self.map_x, self.map_y])
        self.step_counter = 0
        self.trajectories = {i: [] for i in range(self.number_of_robots)}

        # 重置机器人位置
        self.restart_environment.reset_robots()
        self.updateRobotPosition()

        # 第一次调用reset函数时，需要初始化目标实体
        if self.initGoal:
            # 生成目标
            self.current_goal_location = self.restart_environment.spawn_goal()
            self.initGoal = False

        # 如果机器人达到目标的次数超过50次，则更改目标位置
        if self.reached_goal_counter > 50:
            # time.sleep(2)
            msg = 'Found Goal, The robots have found the goal: ' + str(self.total_goal_counter) + ' times'
            self.logger.log(msg)

            self.current_goal_location = self.restart_environment.move_goal()
            self.reached_goal_counter = 0

        # 这段函数当前没有意义（可以将其添加到观测空间或奖励中，但这会改变仅依赖于激光雷达的整个概念）
        self.goal_distance = self.getGoalDistace()

        # 从所有机器人读取激光雷达扫描数据
        robot_scans = []
        for i in range(self.number_of_robots):
            data = None
            scan_data = self.scan_subscriber_list[i]

            while data is None:
                rclpy.spin_once(scan_data)
                # scan_data.get_logger().info("Reading data")
                data = scan_data.scan
            robot_scans.append(data)

        # 将所有激光雷达扫描数据传递给getState函数，它会调整激光雷达数据的大小并告诉我们回合是否结束或截断
        resized_scans = []
        print('debug1')
        for i in range(self.number_of_robots):
            rscan = robot_scans[i]
            lidar_data = self.resize_lidar(rscan)
            resized_scans.append(lidar_data)

        # 每个机器人的观测字典
        robot_observations = {}

        obs = self.addVelocitiesToObs(resized_scans)
        for i, val in enumerate(obs):
            robot_observations['robot' + str(i)] = val

        # 重置机器人距离
        self.distance = [10,10,10]
        
        return robot_observations


# 发布速度指令的类
class PublishCMD_VEL(Node):
    def __init__(self, robot_number):
        super().__init__('PublishCMD_VEL'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/cmd_vel"

        self.cmd_vel_publisher = self.create_publisher(
            Twist, topic_name, 10)
        self.cmd_vel = ' '

    def pub_vel(self):
        self.cmd_vel_publisher.publish(self.cmd_vel)

# 读取里程计数据的类
class ReadOdom(Node):
    def __init__(self, robot_number):
        super().__init__('ReadOdom'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/odom"
        self.subscriber = self.create_subscription(
            Odometry, topic_name, self.odom_callback, 10)
        self.odom = None

    def odom_callback(self, data):
        self.odom = data

# 读取激光扫描数据的类
class ReadScan(Node):
    def __init__(self, robot_number):
        super().__init__('ReadScan'+str(robot_number))
        topic_name = "/my_bot"+str(robot_number)+"/scan"
        self.subscriber = self.create_subscription(LaserScan, topic_name, self.scan_callback,
                                                   qos_profile=qos_profile_sensor_data)
        self.scan = None

    def scan_callback(self, data):
        self.scan = data

# 发布消息的类
class Logger(Node):
    def __init__(self):
        super().__init__('logger')
        
    def log(self, string):
        self.get_logger().info(string)


        


