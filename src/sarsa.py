#! /usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from std_msgs.msg import Float32MultiArray, Float32
from environment import Env


EPISODES = 3000

class Sarsa:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_ml/scripts/sarsa', 'turtlebot3_ml/saved_model/sarsa/stage_2_')
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

if __name__ == '__main__':
    rospy.init_node('sarsa_turtlebot3')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    # pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    # get_action = Float32MultiArray()

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
    try:
        for e in range(total_episodes):
            done = False

            score = 0. #Should going forward give more reward then L/R ?

            state = env.reset()

            if agent.epsilon > 0.05:
                agent.epsilon *= epsilon_discount

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
                    rospy.loginfo ("Time out")
                    break

                if not(done):
                    state = next_state
                else:
                    # last_time_steps = np.append(last_time_steps, [int(i + 1)])
                    result.data = [score, np.max(agent.q)]
                    pub_result.publish(score)
                    score_list.append(score)
                    epsilon_list.append(e)

                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo("EP: "+str(e+1)+" - [alpha: "+str(round(agent.alpha,2))+" - gamma: "+str(round(agent.gamma,2))+" - epsilon: "+str(round(agent.epsilon,2))+"] - Reward: "+str(score)+" \
                            Time: %d:%02d:%02d" % (h, m, s))
                    param_keys = ['epsilon']
                    param_values = [agent.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    score_list.append(score)
                    epsilon_list.append(agent.epsilon)
                    break

            if e%10==0:
                # agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                        json.dump(param_dictionary, outfile)

    except KeyboardInterrupt as e:
        rospy.loginfo("Stopping Turtlebot")
        rospy.loginfo("Writing values to file")
        writeData(score_list, "./data/sarsa_scores")
        writeData(epsilon_list, "./data/sarsa_epsilon")
        rospy.loginfo("Done. Exiting")