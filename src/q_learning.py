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

from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Dropout, Activation

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

class qLearnAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        if rospy.get_param("/use_hitl"):
            self.dirPath = self.dirPath.replace('turtlebot3_rl/src', 'turtlebot3_rl/saved_model/q_learning_hitl/model_')
        else:
            self.dirPath = self.dirPath.replace('turtlebot3_rl/src', 'turtlebot3_rl/saved_model/q_learning/model_')
        self.result = Float32MultiArray()

        self.load_model = rospy.get_param("/load_model", False)
        self.load_episode = rospy.get_param("/load_episode", 0)
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath + str(self.load_episode) + ".h5").get_weights())

            with open(self.dirPath + str(self.load_episode) + '.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    def buildModel(self):
        model = Sequential()
        dropout = 0.2

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))

        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))

        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()

        return model

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        # rospy.loginfo("Updated TARGET NETWORK!!!")
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.float64)
        Y_batch = np.empty((0, self.action_size), dtype=np.float64)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states.copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.action_size]), axis=0)

        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

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
            agent.updateTargetModel()

    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    rospy.init_node('q_learning_turtlebot3')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    use_hitl = rospy.get_param("/use_hitl", False)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

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

    agent = qLearnAgent(state_size, action_size)
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    if use_hitl:
        print(msg)
    
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        
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

        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            agent.appendMemory(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                if global_step <= agent.target_update:
                    agent.trainModel()
                else:
                    agent.trainModel(True)

            score += reward
            state = next_state
            
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            if t > 500:
                # rospy.loginfo("Time out.")
                done = True

            if e % 10 == 0:
                agent.model.save(agent.dirPath + str(e) + '.h5')
                with open(agent.dirPath + str(e) + '.json', 'w') as outfile:
                    json.dump(param_dictionary, outfile)



            if done or env.get_goalbox:
                result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                if e% 10 == 0:
                    agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                print ('-'*100)
                if use_hitl:
                    rospy.loginfo("Q-LEARNING NETWORK ASSISTED HITL!!")
                else:
                    rospy.loginfo("Q-LEARNING NETWORK!!!")

                rospy.loginfo('Ep: %d Q value: %.2f Reward %.2f epsilon: %.2f time: %d:%02d:%02d',
                            e, float(np.max(agent.q_value)), score, agent.epsilon, h, m, s)

                if t> 500:
                    rospy.loginfo("TIME OUT!!!")
                elif done and not env.get_goalbox:
                    rospy.loginfo("COLLISION OCCURED!!!")
                    done = False
                elif env.get_goalbox:
                    rospy.loginfo("GOAL REACHED!!!")
                    env.get_goalbox = False
                

                param_keys = ['epsilon', 'q-value']
                param_values = [agent.epsilon, float(np.max(agent.q_value))]
                param_dictionary = dict(zip(param_keys, param_values))
                score_list.append(score)
                epsilon_list.append(agent.epsilon)
                env.reset()
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                agent.updateTargetModel()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay