from __future__ import print_function
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
import msvcrt
import cv2
import sys
import random
import numpy as np
from collections import deque

SCREENWIDTH = 400
SCREENHEIGHT = 400

seee = False
problemrate = 0.998
GAME = 'way'
ACTIONS = 3 # number of valid actions
GAMMA = 0.8 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.9999  # starting value of epsilon
#INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
stepsize = 1
angle = (math.pi/180)*5


class GameState:
    def __init__(self):
        self.x = []
        self.y = []
        self.score = 0
        self.playerx = 1
        self.playery = 1
        self.goalx = SCREENWIDTH - 2
        self.goaly = SCREENHEIGHT - 2
        self.map = np.zeros((SCREENWIDTH, SCREENHEIGHT))
        self.ori = 0
        [rows, cols] = self.map.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                if self.playerx - 1<= i <= self.playerx + 1 and self.playery - 1<= j <= self.playery + 1:
                    self.map[i, j] = 0
                elif self.goalx - 1<= i <= self.goalx + 1 and self.goaly - 1<= j <= self.goaly + 1:
                    self.map[i, j] = 0
                    if self.goalx == i and self.goaly == j:
                        self.map[i, j] = 1
                else:
                    if random.random() > problemrate:
                        self.map[i, j] = -1
                        self.x.append(i)
                        self.y.append(j)
                    else:
                        self.map[i, j] = 0
        if seee: self.printt()

    def frame_step(self, input_actions):
        terminal = False
        a = 0
        predis = math.sqrt(((self.playerx - self.goalx) * (self.playerx - self.goalx)) + (
                (self.playery - self.goaly) * (self.playery - self.goaly)))
        if sum(input_actions) != 1:
            print("Wrong Action")
        elif input_actions[0] == 1:
            tori = self.ori
            if tori > 2 * math.pi : tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.playerx + (stepsize * math.cos(tori))
            ty = self.playery + (stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH-1 and 0 <= ty <= SCREENHEIGHT-1:
                self.ori = tori
                self.playerx = tx
                self.playery = ty
                tx = math.floor(self.playerx)
                ty = math.floor(self.playery)
                if self.map[tx,ty] == -1 or self.map[tx+1,ty] == -1 or self.map[tx,ty+1] == -1 or self.map[tx+1,ty+1] == -1:
                    terminal = True
                    a = -1
                elif self.map[tx,ty] == 1 or self.map[tx+1,ty] == 1 or self.map[tx,ty+1] == 1 or self.map[tx+1,ty+1] == 1:
                    terminal = True
                    a = 2
                else:
                    a = 0
            else:
                terminal = True
                a = -1
        elif input_actions[1] == 1:
            tori = self.ori + angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0 : tori = tori + 2 * math.pi
            tx = self.playerx + (stepsize * math.cos(tori))
            ty = self.playery + (stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.ori = tori
                self.playerx = tx
                self.playery = ty
                tx = math.floor(self.playerx)
                ty = math.floor(self.playery)
                if self.map[tx, ty] == -1 or self.map[tx + 1, ty] == -1 or self.map[tx, ty + 1] == -1 or self.map[
                    tx + 1, ty + 1] == -1:
                    terminal = True
                    a = -1
                elif self.map[tx, ty] == 1 or self.map[tx + 1, ty] == 1 or self.map[tx, ty + 1] == 1 or self.map[
                    tx + 1, ty + 1] == 1:
                    terminal = True
                    a = 2
                else:
                    a = 0
            else:
                terminal = True
                a = -1
        else:
            tori = self.ori - angle
            if tori > 2 * math.pi: tori = tori - 2 * math.pi
            if tori < 0: tori = tori + 2 * math.pi
            tx = self.playerx + (stepsize * math.cos(tori))
            ty = self.playery + (stepsize * math.sin(tori))
            if 0 <= tx <= SCREENWIDTH - 1 and 0 <= ty <= SCREENHEIGHT - 1:
                self.ori = tori
                self.playerx = tx
                self.playery = ty
                tx = math.floor(self.playerx)
                ty = math.floor(self.playery)
                if self.map[tx, ty] == -1 or self.map[tx + 1, ty] == -1 or self.map[tx, ty + 1] == -1 or self.map[
                    tx + 1, ty + 1] == -1:
                    terminal = True
                    a = -1
                elif self.map[tx, ty] == 1 or self.map[tx + 1, ty] == 1 or self.map[tx, ty + 1] == 1 or self.map[
                    tx + 1, ty + 1] == 1:
                    terminal = True
                    a = 2
                else:
                    a = 0
            else:
                terminal = True
                a = -1
        dis = math.sqrt(((self.playerx - self.goalx) * (self.playerx - self.goalx)) + (
                (self.playery - self.goaly) * (self.playery - self.goaly)))
        if a == 0 : a = (predis - dis)/10
        if terminal:
            del self.x[:]
            del self.y[:]
            self.score = 0
            self.playerx = 1
            self.playery = 1
            self.goalx = SCREENWIDTH - 2
            self.goaly = SCREENHEIGHT - 2
            self.map = np.zeros((SCREENWIDTH, SCREENHEIGHT))
            self.ori = 0
            [rows, cols] = self.map.shape
            for i in range(rows - 1):
                for j in range(cols - 1):
                    if self.playerx - 1 <= i <= self.playerx + 1 and self.playery - 1 <= j <= self.playery + 1:
                        self.map[i, j] = 0
                    elif self.goalx - 1 <= i <= self.goalx + 1 and self.goaly - 1 <= j <= self.goaly + 1:
                        self.map[i, j] = 0
                        if self.goalx == i and self.goaly  == j:
                            self.map[i, j] = 1
                    else:
                        if random.random() > problemrate:
                            self.map[i, j] = -1
                            self.x.append(i)
                            self.y.append(j)
                        else:
                            self.map[i, j] = 0
        if seee: self.printt()
        submap = np.array(bytearray(os.urandom(6400)))
        submap = submap.reshape(80, 80)
        [rows, cols] = submap.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                ii = math.floor(self.playerx) - 39 + i
                jj = math.floor(self.playery) - 39 + j
                if ii < 0 or jj < 0 or ii > SCREENWIDTH - 1 or jj > SCREENHEIGHT - 1 or self.map[ii, jj] == -1:
                    submap[i, j] = 0
                elif self.map[ii, jj] == 1:
                    submap[i, j] = 255
                else:
                    # submap[i, j] = 1
                    submap[i, j] = 125
        return submap, a, terminal

    def printt(self):
        plt.clf()
        #print(self.map);
        colors1 = '#FF0000'  # 点的颜色
        colors2 = '#0000FF'
        area = 0.1  # 点面积
        # 画散点图
        plt.scatter(self.x, self.y, s=area, c=colors1)
        plt.scatter(self.playerx, self.playery, s=area, c=colors2)
        plt.scatter(self.goalx, self.goaly, s=area, c=colors2)
        plt.xlabel("x")
        plt.ylabel('y')
        plt.title("title")
        plt.axis('equal')
        plt.xlim(-5, 405)
        plt.ylim(-5, 405)
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

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply  (readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    #a_file = open("logs_" + GAME + "/readout.txt", 'w')
    #h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    #ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved2_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        if msvcrt.kbhit() : input()
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        #x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        #ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1_colored, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch}
            )

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved2_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
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
    sess = tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)
    if seee : plt.ioff()
    if seee : plt.show()

def main():
    playGame()


if __name__ == "__main__":
    main()