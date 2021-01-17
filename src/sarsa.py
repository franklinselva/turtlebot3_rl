#! /usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
from collections import deque
import select, termios, tty


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from std_msgs.msg import Float32MultiArray
from environment import Env
from hitl import PublishThread, vels


EPISODES = 3000


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

class Sarsa:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        if rospy.get_param("/use_hitl"):
            self.dirPath = self.dirPath.replace('turtlebot3_rl/src', 'turtlebot3_rl/saved_model/sarsa_hitl/model_')
        else:
            self.dirPath = self.dirPath.replace('turtlebot3_rl/src', 'turtlebot3_rl/saved_model/sarsa/model_')
        self.result = Float32MultiArray()

        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQvalue(self, state, action):
        return self.q.get(tuple(state), 0.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q.get(tuple(state), None)
        if oldv is None:
            self.q[tuple(state)] = reward 
        else:
            self.q[tuple(state)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQvalue(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2, action2):
        qnext = self.getQvalue(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)

def writeData(data, filename):
    '''
    Write list to pickle file
    '''
    import pickle

    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)



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
            # agent.updateTargetModel()

    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    rospy.init_node('sarsa_turtlebot3')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    use_hitl = rospy.get_param("/use_hitl", False)

    # pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    # get_action = Float32MultiArray()

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


    state_size = 364 # No.of data from laser scan
    action_size = 5 #Left, Right, Rotate Front and Back

    # List for saving data to file
    score_list = []
    epsilon_list = []

    env = Env(action_size)

    agent = Sarsa(actions = range(action_size), epsilon=0.9, alpha=0.2, gamma=0.9)

    initial_epsilon = agent.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 10000
    step_size = 1500
    highest_reward = 0.

    if use_hitl:
        print(msg)

    try:
        for e in range(total_episodes):
            done = False

            score = 0. #Should going forward give more reward then L/R ?

            state = env.reset()

            if agent.epsilon > 0.05:
                agent.epsilon *= epsilon_discount

            if use_hitl:
                print (env.collision)
                if env.collision % 5 == 0 and env.collision != 0:
                
                    rospy.loginfo("WAITING FOR HUMAN FEEDBACK!!!")
                    print(vels(speed,turn))
                    publishHITL(pub_thread, agent, False)
                    env.collision += 1 # Counting as collision for human feedback ##BUG##
                else:
                    publishHITL(pub_thread, agent, True)

            for i in range(step_size):

                # Pick an action based on the current state
                action = agent.chooseAction(state)

                # Execute the action and get feedback
                next_state, reward, done = env.step(action)
                score += reward

                if highest_reward < score:
                    highest_reward = score

                next_action = agent.chooseAction(next_state)

                #sarsa.learn(state, action, reward, nextState)
                agent.learn(state, action, reward, next_state, next_action)

                # env._flush(force=True)
                if i > 750: 
                    # rospy.loginfo ("Time out")
                    break
        
                if done or env.get_goalbox:
                    # last_time_steps = np.append(last_time_steps, [int(i + 1)])
                    result.data = [score, np.mean(agent.q.values())]
                    pub_result.publish(result)
                    score_list.append(score)
                    epsilon_list.append(e)

                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    if use_hitl:
                        rospy.loginfo("SARSA ASSISTED WITH HITL!!!")
                    else:
                        rospy.loginfo("SARSA ALGORITHM!!!")


                    rospy.loginfo("EP: %d Q-value: %.2f Epsilon: %.2f Reward: %.2f Time: %d:%02d:%02d", e+1, np.mean(agent.q.values()), agent.epsilon, \
                        score, h, m, s)

                    if i > 740:
                        rospy.loginfo("TIME OUT!!!")
                    elif done and not env.get_goalbox:
                        rospy.loginfo("COLLISION OCCURED!!!")
                        done = False
                    elif env.get_goalbox:
                        rospy.loginfo("GOAL REACHED!!!")
                        env.get_goalbox = False
                        
                    print ('-'*100)
                    param_keys = ['epsilon', 'q-value']
                    param_values = [agent.epsilon, float(np.mean(agent.q.values()))]
                    param_dictionary = dict(zip(param_keys, param_values))
                    score_list.append(score)
                    epsilon_list.append(agent.epsilon)
                    break
                else:
                    state = next_state

            if (e+1)%10==0:
                # agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e+1) + '.json', 'w') as outfile:
                        json.dump(param_dictionary, outfile)

    except KeyboardInterrupt as e:
        rospy.loginfo("Stopping Turtlebot")
        rospy.loginfo("Writing values to file")
        # writeData(score_list, "./data/sarsa_scores")
        # writeData(epsilon_list, "./data/sarsa_epsilon")
        rospy.loginfo("Done. Exiting")