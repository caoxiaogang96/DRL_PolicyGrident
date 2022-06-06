import numpy as np
import gym
from itertools import count
import torch
import argparse

from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNet(nn.Module):
    
    rewards = []
    action_logs = []
    
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        
       #self.fc1 = nn.Linear(input_dim, 24)
       #self.fc2 = nn.Linear(24, 36)
       #self.fc3 = nn.Linear(36, output_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, output_dim)
    
        self.optimizer = optim.Adam(self.parameters(),lr=1e-2)


    def forward(self, x):
        #x = F.relu(self.fc1(x))
        #print("relu:", x)
        #x = F.softmax(self.fc2(x), dim=1)
        #print("softmax:", x)
        #return x
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)
    
    def gen_sample(self, args):
        reward_batch = []
        for i_episode in range(0, args.batch):
            reward_log = [] # 记录这一幕的受益轨迹
            action_log = [] # 记录这一幕的动作对数概率轨迹
            state, ep_reward = env.reset(), 0  # 初始化当前状态 新的一幕
            for i in range(0, args.max_steps): #最多执行1000步
                state = torch.from_numpy(state).float().unsqueeze(0) # 将numpy转成torch格式
                actions = self(state)     # 根据当前状态计算各个动作的评分/概率
                #print(i, self.action_log)
                m = Categorical(actions)
                action = m.sample() # 根据概率采样
                next_state, reward, done, _ = env.step(action.item()) # 执行动作后，状态变化，奖励，以及是否结束
                ep_reward += reward               # 在状态s执行动作a的奖励
                reward_log.append(reward)
                action_log.append(m.log_prob(action))
                
                state = next_state #下一个状态

                if done:
                    self.rewards.append(reward_log)
                    self.action_logs.append(action_log)
                    reward_batch.append(ep_reward)
                    break
                
        return reward_batch
    
    def policy_update(self, args):
        loss = 0
        for rewards, action_logs in zip(self.rewards, self.action_logs):
            #(1)计算每一步收益的贝尔曼方程
            rewards_BM = []
            R = 0
            for reward in rewards[::-1]:
                R = reward + args.gamma * R #就是Q^
                rewards_BM.insert(0, R)

            rewards_BM = torch.tensor(rewards_BM).float() # 转成tensor
            if(args.regular):
                rewards_BM = (rewards_BM - rewards_BM.mean()) / rewards_BM.std() #正则化

            #(2) 计算梯度
            policy_reward = []
            for action_log, R in zip(action_logs, rewards_BM):
                policy_reward.append(action_log * R)

            policy_grad = torch.cat(policy_reward).sum()
            loss = loss + (-1 * policy_grad)

        
        loss = loss / args.batch
        #(3) 利用自动差分器，将负的收益目标当作损失计算
        self.optimizer.zero_grad()
        loss.backward()

        #(4) 更新梯度网络
        self.optimizer.step()

        del self.rewards[:]          # 清空episode 数据
        del self.action_logs[:]
        

def parse():
    parser = argparse.ArgumentParser(description='Pytorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',help='discount factor(default:0.99)')
    parser.add_argument('--seed',type=int, default=543, metavar='N',help='random seed (default: 543)')
    parser.add_argument('--batch', type=int, default=1, help='batch number（default: 1）')
    parser.add_argument('--regular', type=int, default=1, help='Regularization（default: True）')
    parser.add_argument('--episode', type=int, default=10000, help='max episode（default: 10000）')
    parser.add_argument('--max-steps',type=int, default=1000,help='max step number（default: 1000）')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    
    return args

if __name__ == "__main__":
    avg_reward = 10
    sample_cnt = 0
    avg_reward_log = []
    ep_reward_log = []
          
    args = parse()
    env = gym.make('CartPole-v1')
    env.seed(args.seed)
    torch.manual_seed(args.seed)    # 策略梯度算法方差很大，设置seed以保证复现性

    state_n = env.observation_space.shape[0] #状态维度
    action_n = env.action_space.n #动作空间

    policy = PolicyNet(state_n, action_n)
    
    for i_episode in range(args.episode):
        #（1）生成样本
        ep_rewards = policy.gen_sample(args)
        sample_cnt += 1
        for ep_reward in ep_rewards:
            avg_reward = 0.05 * ep_reward + 0.95 * avg_reward
            avg_reward_log.append(avg_reward)
            ep_reward_log.append(ep_reward) 

        # （2）计算梯度
        # （3）更新网络
        policy.policy_update(args)

        #（4） 打印信息
        if((i_episode % args.log_interval) == 0):
             print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, avg_reward))

        if avg_reward > env.spec.reward_threshold:   # 大于游戏的最大阈值475时，退出游戏
                print("Solved! Running reward is now {}!".format(avg_reward))
                break
    
    plt.plot(range(0, len(avg_reward_log)), avg_reward_log, label="reward")
    plt.show()
    