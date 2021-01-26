#!/usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import copy


import random
import time
import sys
from collections import deque
import select, termios, tty

from hitl import PublishThread, vels
from .utils.function import soft_update, hard_update
from .utils.onu_noise import OUNoise
from environment import Env

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


#---Run agent---#

is_training = True

exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 750
MAX_BUFFER = 200000
rewards_all_episodes = []

STATE_DIMENSION = 364
ACTION_DIMENSION = 5
ACTION_V_MAX = 0.25 # m/s
ACTION_W_MAX = 2. # rad/s


BATCH_SIZE = 100
LEARNING_RATE = 0.0003
GAMMA = 0.99
TAU = 0.001

if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
    
x = 0
y = 0
z = 0
th = 0
status = 0
key_timeout = None
speed = None
repeat = None
turn = None

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0,0,0),
        'o':(1,0,0,-1),
        'j':(0,0,0,1),
        'l':(0,0,0,-1),
        'u':(1,0,0,1),
        ',':(-1,0,0,0),
        '.':(-1,0,0,1),
        'm':(-1,0,0,-1),
        'O':(1,-1,0,0),
        'I':(1,0,0,0),
        'J':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(1,1,0,0),
        '<':(-1,0,0,0),
        '>':(-1,-1,0,0),
        'M':(-1,1,0,0),
        't':(0,0,1,0),
        'b':(0,0,-1,0),
    }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
    }

#---Critic--#

EPS = 0.003

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 364)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)
        
    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.softmax(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.softmax(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])
        
        return s_array, a_array, r_array, new_s_array, done_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        #self.iter = 0 
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()
        
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        noise = self.noise.sample()
        # print('noisea', noise)
        noise[0] = noise[0]*self.action_limit_v
        noise[1] = noise[1]*self.action_limit_w
        # print('noise', noise)
        new_action = action.data.numpy() #+ noise
        #print('action_no', new_action)
        return new_action
    
    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)
        
        #-------------- optimize critic
        
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + (1 - done_sample)*GAMMA*next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        #-------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        #print(self.qvalue, torch.max(self.qvalue))
        #----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
    
    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), dirPath +'/Models/' + '/' + str(episode_count)+ '_actor.pt')
        torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + '/'+str(episode_count)+ '_critic.pt')
        print('****Models saved***')
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dirPath + '/Models/' + '/'+str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dirPath + '/Models/' + '/'+str(episode)+ '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')



print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
noise = OUNoise(ACTION_DIMENSION, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
trainer.load_models(560)


def getKey(key_timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], key_timeout)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def publishHITL(pub_thread, agent, Done = False):
    '''
    Function to call publish thread for teleop key
    '''
    global x, y, z, th, speed, turn, key_timeout, status, settings
    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(x, y, z, th, speed, turn)

        while  not Done:
            # action = agent.getAction()

            #agent.appendMemory(state, action, )
            key = getKey(key_timeout)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                turn = turn * speedBindings[key][1]

                print(vels(speed,turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key=='' and  x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break
 
            pub_thread.update(x, y, z, th, speed, turn)
            agent.updateTargetModel()

    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    rospy.init_node('sac_discrete')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    use_hitl = rospy.get_param("/use_hitl", False)
    result = Float32MultiArray()
    env = Env(action_dim=ACTION_DIMENSION)
    before_training = 4

    past_action = np.zeros(ACTION_DIMENSION)
    rewards_current_episode = 0.
    
    global settings
    settings = termios.tcgetattr(sys.stdin)
    
    if use_hitl:
        speed = rospy.get_param("~speed", 0.5)
        turn = rospy.get_param("~turn", 2.0)
        repeat = rospy.get_param("~repeat_rate", 0.0)
        key_timeout = rospy.get_param("~key_timeout", 0.1)
        if key_timeout == 0.0:
            key_timeout = None

        pub_thread = PublishThread(repeat)
    
    for ep in range(MAX_EPISODES):
        done = False
        state = env.reset()
        if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')

        if use_hitl:
            # print (env.collision)
            if env.collision % 5 == 0 and env.collision != 0:
                
                rospy.loginfo("WAITING FOR HUMAN FEEDBACK!!!")
                print(vels(speed,turn))
                publishHITL(pub_thread, agent, False)
                env.collision += 1 # Counting as collision for human feedback ##BUG##
                # env.setReward(state, False, action)
            else:
                publishHITL(pub_thread, agent, True)
        
        for step in range(MAX_STEPS):
            state = np.float32(state)

            # if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
            if is_training and not ep%10 == 0:
                action = trainer.get_exploration_action(state)
                
                N = copy.deepcopy(noise.get_noise(t=step))
                N[0] = N[0]*ACTION_V_MAX/2
                N[1] = N[1]*ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
            else:
                action = trainer.get_exploration_action(state)

            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done = env.step(action, past_action)
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)
            
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep%10 == 0 or not ram.len >= before_training*MAX_STEPS:
                if reward == 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        ram.add(state, action, reward, next_state, done)
                # elif reward == -100.:
                #     print('***\n-------- Collision ----------\n****')
                #     for _ in range():
                #         ram.add(state, action, reward, next_state, done)
                else:
                    ram.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            

            if ram.len >= before_training*MAX_STEPS and is_training and not ep%10 == 0:
                trainer.optimizer()

            if done or step == MAX_STEPS-1:
                print('reward per ep: ' + str(rewards_current_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                # print('explore_v: ' + str(var_v) + ' and explore_w: ' + str(var_w))
                print('sigma: ' + str(noise.sigma))
                # rewards_all_episodes.append(rewards_current_episode)

                result = rewards_current_episode
                pub_result.publish(result)

                # if done or collided
                break
        if ep%10 == 0:
            trainer.save_models(ep)

print('Completed Training')
