from __future__ import print_function
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
import random
import numpy as np
from collections import deque

SCREENWIDTH = 80
SCREENHEIGHT = 80

hit = -0.5
rat = 15
crat = 1
staytime = 100
gps=0.1
gc=-5
seee = False
problemrate = 0.998
GAME = 'way'
ACTIONS = 3 # number of valid actions
GAMMA = 0.8 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 5000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.9999  # starting value of epsilon
#INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
e_stepsize = 0.5
e_angle = (math.pi/180)*10
p_stepsize = 0.8
p_angle = (math.pi/180)*5

def safe(i,j):
    if (i-40)*(i-40)+(j-40)*(j-40) < rat*rat:
        return True
    else:
        return False
def dis(i,j,ii,jj):
    return math.sqrt((i-ii)*(i-ii)+(j-jj)*(j-jj))


class GameState:
    def __init__(self):
        self.ex = 40
        self.ey = 20
        self.eori = (math.pi/180)*90
        self.timer = 0
        self.p1x = 20
        self.p1y = 10
        self.p1ori = (math.pi/180)*90
        self.p2x = 40
        self.p2y = 10
        self.p2ori = (math.pi/180)*135
        self.p3x = 60
        self.p3y = 10
        self.p3ori = (math.pi/180)*90
        if seee: self.printt()

    def frame_step(self, input_e_actions,input_p1_actions,input_p2_actions,input_p3_actions):
        flag = 0
        if sum(input_e_actions) != 1:
            print("Wrong Action")
        elif input_e_actions[0] == 1:
            tori = self.eori
            if tori > 2 * math.pi : tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.ex + (e_stepsize * math.cos(tori))
            ty = self.ey + (e_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH-1 and 0 <= ty <= SCREENHEIGHT-1:
                self.eori = tori
                self.ex = tx
                self.ey = ty
            else:
                flag=1
        elif input_e_actions[1] == 1:
            tori = self.eori + e_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.ex + (e_stepsize * math.cos(tori))
            ty = self.ey + (e_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.eori = tori
                self.ex = tx
                self.ey = ty
            else:
                flag=1
        else:
            tori = self.eori - e_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.ex + (e_stepsize * math.cos(tori))
            ty = self.ey + (e_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.eori = tori
                self.ex = tx
                self.ey = ty
            else:
                flag=1

        if sum(input_p1_actions) != 1:
            print("Wrong Action")
        elif input_p1_actions[0] == 1:
            tori = self.p1ori
            if tori > 2 * math.pi : tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.p1x + (p_stepsize * math.cos(tori))
            ty = self.p1y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH-1 and 0 <= ty <= SCREENHEIGHT-1:
                self.p1ori = tori
                self.p1x = tx
                self.p1y = ty
            else:
                flag=2
        elif input_p1_actions[1] == 1:
            tori = self.p1ori + p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p1x + (p_stepsize * math.cos(tori))
            ty = self.p1y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p1ori = tori
                self.p1x = tx
                self.p1y = ty
            else:
                flag=2
        else:
            tori = self.p1ori - p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p1x + (p_stepsize * math.cos(tori))
            ty = self.p1y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p1ori = tori
                self.p1x = tx
                self.p1y = ty
            else:
                flag=2

        if sum(input_p2_actions) != 1:
            print("Wrong Action")
        elif input_p2_actions[0] == 1:
            tori = self.p2ori
            if tori > 2 * math.pi : tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.p2x + (p_stepsize * math.cos(tori))
            ty = self.p2y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH-1 and 0 <= ty <= SCREENHEIGHT-1:
                self.p2ori = tori
                self.p2x = tx
                self.p2y = ty
            else:
                flag=2
        elif input_p2_actions[1] == 1:
            tori = self.p2ori + p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p2x + (p_stepsize * math.cos(tori))
            ty = self.p2y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p2ori = tori
                self.p2x = tx
                self.p2y = ty
            else:
                flag=2
        else:
            tori = self.p2ori - p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p2x + (p_stepsize * math.cos(tori))
            ty = self.p2y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p2ori = tori
                self.p2x = tx
                self.p2y = ty
            else:
                flag=2

        if sum(input_p3_actions) != 1:
            print("Wrong Action")
        elif input_p3_actions[0] == 1:
            tori = self.p3ori
            if tori > 2 * math.pi : tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.p3x + (p_stepsize * math.cos(tori))
            ty = self.p3y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH-1 and 0 <= ty <= SCREENHEIGHT-1:
                self.p3ori = tori
                self.p3x = tx
                self.p3y = ty
            else:
                flag=2
        elif input_p3_actions[1] == 1:
            tori = self.p3ori + p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p3x + (p_stepsize * math.cos(tori))
            ty = self.p3y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p3ori = tori
                self.p3x = tx
                self.p3y = ty
            else:
                flag=2
        else:
            tori = self.p3ori - p_angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.p3x + (p_stepsize * math.cos(tori))
            ty = self.p3y + (p_stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.p3ori = tori
                self.p3x = tx
                self.p3y = ty
            else:
                flag=2
        if safe(self.ex,self.ey):
            self.timer += 1
        else:
            if self.timer > 0 : self.timer -= 1

        if self.timer > staytime:
            terminal=True
            a = gc
        elif dis(self.ex,self.ey,self.p1x,self.p1y)<crat:
            terminal = True
            a = gc
        elif dis(self.ex,self.ey,self.p2x,self.p2y)<crat:
            terminal = True
            a = gc
        elif dis(self.ex,self.ey,self.p3x,self.p3y)<crat:
            terminal = True
            a = gc
        elif safe(self.p1x,self.p1y):
            terminal = True
            a = -gc
        elif safe(self.p2x,self.p2y):
            terminal = True
            a = -gc
        elif safe(self.p3x,self.p3y):
            terminal = True
            a = -gc
        elif flag == 1 :
            terminal = True
            a = hit
        elif flag == 2:
            terminal = True
            a = -hit
        else:
            terminal = False
            a = gps


        if terminal:
            self.ex = 40
            self.ey = 20
            self.eori = (math.pi / 180) * 90
            self.timer = 0
            self.p1x = 20
            self.p1y = 10
            self.p1ori = (math.pi / 180) * 90
            self.p2x = 40
            self.p2y = 10
            self.p2ori = (math.pi / 180) * 135
            self.p3x = 60
            self.p3y = 10
            self.p3ori = (math.pi / 180) * 90

        submap1 = np.array(bytearray(os.urandom(SCREENWIDTH*SCREENHEIGHT)))
        submap1 = submap1.reshape(SCREENWIDTH, SCREENHEIGHT)
        for i in range(SCREENWIDTH ):
            for j in range(SCREENHEIGHT ):
                if safe(i,j):
                    submap1[i, j] = 100
                elif math.floor(self.ex)==i and math.floor(self.ey)==j:
                    submap1[i, j] = self.timer
                elif math.floor(self.p1x)==i and math.floor(self.p1y)==j:
                    submap1[i, j] = 255
                elif math.floor(self.p2x)==i and math.floor(self.p2y)==j:
                    submap1[i, j] = 225
                elif math.floor(self.p3x)==i and math.floor(self.p3y)==j:
                    submap1[i, j] = 175
                else:
                    # submap[i, j] = 1
                    submap1[i, j] = 125

        submap2 = np.array(bytearray(os.urandom(SCREENWIDTH * SCREENHEIGHT)))
        submap2 = submap2.reshape(SCREENWIDTH, SCREENHEIGHT)
        for i in range(SCREENWIDTH ):
            for j in range(SCREENHEIGHT ):
                if safe(i, j):
                    submap2[i, j] = 100
                elif math.floor(self.ex) == i and math.floor(self.ey) == j and (not safe(i, j)):
                    submap2[i, j] = self.timer
                elif math.floor(self.p1x) == i and math.floor(self.p1y) == j:
                    submap2[i, j] = 255
                elif math.floor(self.p2x) == i and math.floor(self.p2y) == j:
                    submap2[i, j] = 225
                elif math.floor(self.p3x) == i and math.floor(self.p3y) == j:
                    submap2[i, j] = 175
                else:
                    # submap[i, j] = 1
                    submap1[i, j] = 125
        if seee: self.printt()
        return submap1,submap2, a, terminal

    def printt(self):
        plt.clf()
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = 40 + rat * np.cos(theta)
        y = 40 + rat * np.sin(theta)
        #print(self.map);
        colors1 = '#FF0000'  # 红点的颜色
        colors2 = '#0000FF'  #蓝
        colors3 = '#A9A9A9' #黄
        area = 0.7  # 点面积
        # 画散点图
        plt.scatter(x, y, s=area,c=colors3)
        plt.scatter(self.p1x, self.p1y, s=area, c=colors1)
        plt.scatter(self.p2x, self.p2y, s=area, c=colors1)
        plt.scatter(self.p3x, self.p3y, s=area, c=colors1)
        plt.scatter(self.ex, self.ey, s=area, c=colors2)
        plt.xlabel("x")
        plt.ylabel('y')
        plt.title('%d' % self.timer)
        plt.axis('equal')
        plt.xlim(-5, SCREENWIDTH+10)
        plt.ylim(-5, SCREENHEIGHT+10)
        plt.pause(0.0001)
        #plt.clf()  # 清图。
        #plt.cla()  # 清坐标轴。
        #plt.close()  # 关窗口


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s0, readout0, h_fc10, sess0,s1, readout1, h_fc11, sess1,s2, readout2, h_fc12, sess2,s3, readout3, h_fc13, sess3):
    # define the cost function
    a0 = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y0 = tf.compat.v1.placeholder("float", [None])
    readout_action0 = tf.reduce_sum(tf.multiply  (readout0, a0), reduction_indices=1)
    cost0 = tf.reduce_mean(tf.square(y0 - readout_action0))
    train_step0 = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost0)

    a1 = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y1 = tf.compat.v1.placeholder("float", [None])
    readout_action1 = tf.reduce_sum(tf.multiply(readout1, a1), reduction_indices=1)
    cost1 = tf.reduce_mean(tf.square(y1 - readout_action1))
    train_step1 = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost1)

    a2 = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y2 = tf.compat.v1.placeholder("float", [None])
    readout_action2 = tf.reduce_sum(tf.multiply(readout2, a2), reduction_indices=1)
    cost2 = tf.reduce_mean(tf.square(y2 - readout_action2))
    train_step2 = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost2)

    a3 = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y3 = tf.compat.v1.placeholder("float", [None])
    readout_action3 = tf.reduce_sum(tf.multiply(readout3, a3), reduction_indices=1)
    cost3 = tf.reduce_mean(tf.square(y3 - readout_action3))
    train_step3 = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost3)

    # open up a game state to communicate with emulator
    game_state = GameState()

    # store the previous observations in replay memory
    D0 = deque()
    D1 = deque()
    D2 = deque()
    D3 = deque()

    # printing
    #a_file = open("logs_" + GAME + "/readout.txt", 'w')
    #h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t0,x_t1, r_0, terminal = game_state.frame_step(do_nothing, do_nothing, do_nothing, do_nothing)
    #x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    #ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t0 = np.stack((x_t0, x_t0, x_t0, x_t0), axis=2)
    s_t1 = np.stack((x_t1, x_t1, x_t1, x_t1), axis=2)

    # saving and loading networks
    saver0 = tf.compat.v1.train.Saver()
    sess0.run(tf.compat.v1.global_variables_initializer())
    checkpoint0 = tf.train.get_checkpoint_state("saved3_networks/e")
    if checkpoint0 and checkpoint0.model_checkpoint_path:
        saver0.restore(sess0, checkpoint0.model_checkpoint_path)
        print("Successfully loaded:", checkpoint0.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    saver1 = tf.compat.v1.train.Saver()
    sess1.run(tf.compat.v1.global_variables_initializer())
    checkpoint1 = tf.train.get_checkpoint_state("saved3_networks/p1")
    if checkpoint1 and checkpoint1.model_checkpoint_path:
        saver1.restore(sess1, checkpoint1.model_checkpoint_path)
        print("Successfully loaded:", checkpoint1.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    saver2 = tf.compat.v1.train.Saver()
    sess2.run(tf.compat.v1.global_variables_initializer())
    checkpoint2 = tf.train.get_checkpoint_state("saved3_networks/p2")
    if checkpoint2 and checkpoint2.model_checkpoint_path:
        saver2.restore(sess2, checkpoint2.model_checkpoint_path)
        print("Successfully loaded:", checkpoint2.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    saver3 = tf.compat.v1.train.Saver()
    sess3.run(tf.compat.v1.global_variables_initializer())
    checkpoint3 = tf.train.get_checkpoint_state("saved3_networks/p3")
    if checkpoint3 and checkpoint3.model_checkpoint_path:
        saver3.restore(sess3, checkpoint3.model_checkpoint_path)
        print("Successfully loaded:", checkpoint3.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t0 = readout0.eval(feed_dict={s0 : [s_t0]})[0]
        a_t0 = np.zeros([ACTIONS])
        action_index0 = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index0 = random.randrange(ACTIONS)
                a_t0[action_index0] = 1
            else:
                action_index0 = np.argmax(readout_t0)
                a_t0[action_index0] = 1
        else:
            a_t0[0] = 1 # do nothing

        readout_t1 = readout1.eval(feed_dict={s1: [s_t1]})[0]
        a_t1 = np.zeros([ACTIONS])
        action_index1 = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index1 = random.randrange(ACTIONS)
                a_t1[action_index1] = 1
            else:
                action_index1 = np.argmax(readout_t1)
                a_t1[action_index1] = 1
        else:
            a_t1[0] = 1  # do nothing

        readout_t2 = readout2.eval(feed_dict={s2: [s_t1]})[0]
        a_t2 = np.zeros([ACTIONS])
        action_index2 = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index2 = random.randrange(ACTIONS)
                a_t2[action_index2] = 1
            else:
                action_index2 = np.argmax(readout_t2)
                a_t2[action_index2] = 1
        else:
            a_t2[0] = 1  # do nothing

        readout_t3 = readout3.eval(feed_dict={s3: [s_t1]})[0]
        a_t3 = np.zeros([ACTIONS])
        action_index3 = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index3 = random.randrange(ACTIONS)
                a_t3[action_index3] = 1
            else:
                action_index3 = np.argmax(readout_t3)
                a_t3[action_index3] = 1
        else:
            a_t3[0] = 1  # do nothing
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored0,x_t1_colored1, r_t, terminal = game_state.frame_step(a_t0,a_t1,a_t2,a_t3)
        #x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t10 = np.reshape(x_t1_colored0, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t10 = np.append(x_t10, s_t0[:, :, :3], axis=2)

        x_t11 = np.reshape(x_t1_colored1, (80, 80, 1))

        s_t11 = np.append(x_t11, s_t1[:, :, :3], axis=2)

        # store the transition in D
        D0.append((s_t0, a_t0, r_t, s_t10, terminal))
        if len(D0) > REPLAY_MEMORY:
            D0.popleft()

        D1.append((s_t1, a_t1, - r_t, s_t11, terminal))
        if len(D1) > REPLAY_MEMORY:
            D1.popleft()

        D2.append((s_t1, a_t2, -r_t, s_t11, terminal))
        if len(D2) > REPLAY_MEMORY:
            D2.popleft()

        D3.append((s_t1, a_t3, -r_t, s_t11, terminal))
        if len(D3) > REPLAY_MEMORY:
            D3.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch0 = random.sample(D0, BATCH)
            # get the batch variables
            s_j_batch0 = [d[0] for d in minibatch0]
            a_batch0 = [d[1] for d in minibatch0]
            r_batch0 = [d[2] for d in minibatch0]
            s_j1_batch0 = [d[3] for d in minibatch0]

            y_batch0 = []
            readout_j1_batch0 = readout0.eval(feed_dict = {s0 : s_j1_batch0})
            for i in range(0, len(minibatch0)):
                terminal = minibatch0[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch0.append(r_batch0[i])
                else:
                    y_batch0.append(r_batch0[i] + GAMMA * np.max(readout_j1_batch0[i]))

            # perform gradient step
            train_step0.run(feed_dict = {
                y0 : y_batch0,
                a0 : a_batch0,
                s0 : s_j_batch0}
            )

            minibatch1 = random.sample(D1, BATCH)
            # get the batch variables
            s_j_batch1 = [d[0] for d in minibatch1]
            a_batch1 = [d[1] for d in minibatch1]
            r_batch1 = [d[2] for d in minibatch1]
            s_j1_batch1 = [d[3] for d in minibatch1]

            y_batch1 = []
            readout_j1_batch1 = readout1.eval(feed_dict={s1: s_j1_batch1})
            for i in range(0, len(minibatch1)):
                terminal = minibatch1[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch1.append(r_batch1[i])
                else:
                    y_batch1.append(r_batch1[i] + GAMMA * np.max(readout_j1_batch1[i]))

            # perform gradient step
            train_step1.run(feed_dict={
                y1: y_batch1,
                a1: a_batch1,
                s1: s_j_batch1}
            )

            minibatch2 = random.sample(D2, BATCH)
            # get the batch variables
            s_j_batch2 = [d[0] for d in minibatch2]
            a_batch2 = [d[1] for d in minibatch2]
            r_batch2 = [d[2] for d in minibatch2]
            s_j1_batch2 = [d[3] for d in minibatch2]

            y_batch2 = []
            readout_j1_batch2 = readout2.eval(feed_dict={s2: s_j1_batch2})
            for i in range(0, len(minibatch2)):
                terminal = minibatch2[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch2.append(r_batch2[i])
                else:
                    y_batch2.append(r_batch2[i] + GAMMA * np.max(readout_j1_batch2[i]))

            # perform gradient step
            train_step2.run(feed_dict={
                y2: y_batch2,
                a2: a_batch2,
                s2: s_j_batch2}
            )

            minibatch3 = random.sample(D3, BATCH)
            # get the batch variables
            s_j_batch3 = [d[0] for d in minibatch3]
            a_batch3 = [d[1] for d in minibatch3]
            r_batch3 = [d[2] for d in minibatch3]
            s_j1_batch3 = [d[3] for d in minibatch3]

            y_batch3 = []
            readout_j1_batch3 = readout3.eval(feed_dict={s3: s_j1_batch3})
            for i in range(0, len(minibatch3)):
                terminal = minibatch3[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch3.append(r_batch3[i])
                else:
                    y_batch3.append(r_batch3[i] + GAMMA * np.max(readout_j1_batch3[i]))

            # perform gradient step
            train_step3.run(feed_dict={
                y3: y_batch3,
                a3: a_batch3,
                s3: s_j_batch3}
            )

        # update the old values
        s_t0 = s_t10
        s_t1 = s_t11
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver0.save(sess0, 'saved3_networks/e/' + GAME + '-dqn', global_step = t)
            saver1.save(sess1, 'saved3_networks/p1/' + GAME + '-dqn', global_step=t)
            saver2.save(sess2, 'saved3_networks/p2/' + GAME + '-dqn', global_step=t)
            saver3.save(sess3, 'saved3_networks/p3/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION0", action_index0, "/ ACTION1", action_index1, "/ ACTION2", action_index2, "/ ACTION3", action_index3, "/ REWARD", r_t, \
              "/ Q_MAX0 %e" % np.max(readout_t0),"/ Q_MAX1 %e" % np.max(readout_t1),"/ Q_MAX2 %e" % np.max(readout_t2),"/ Q_MAX %e3" % np.max(readout_t3))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''



def playGame():
    if seee : plt.ion()
    if seee : plt.figure(figsize=(10, 10))
    if seee : plt.clf()
    sess0 = tf.compat.v1.InteractiveSession()
    s0, readout0, h_fc10 = createNetwork()
    sess1 = tf.compat.v1.InteractiveSession()
    s1, readout1, h_fc11 = createNetwork()
    sess2 = tf.compat.v1.InteractiveSession()
    s2, readout2, h_fc12 = createNetwork()
    sess3 = tf.compat.v1.InteractiveSession()
    s3, readout3, h_fc13 = createNetwork()
    trainNetwork(s0, readout0, h_fc10, sess0,s1, readout1, h_fc11, sess1,s2, readout2, h_fc12, sess2,s3, readout3, h_fc13, sess3)
    if seee : plt.ioff()
    if seee : plt.show()

def main():
    playGame()


if __name__ == "__main__":
    main()