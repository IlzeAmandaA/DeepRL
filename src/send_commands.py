#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np
import csv
import pickle
import robobo as robobo
import cv2 as cv2
import sys
import signal
import random
import prey as prey
from nn_network.nn import neural_net, LossHistory
import time
import argparse

NUM_SENSORS=6
GAMMA=0.8 #too low goes forward too much
ACTIONS={0:'left', 1:'right', 2:'forward'}
FOOD=0
GAUSSIAN =True

parser = argparse.ArgumentParser(description='Deep Reinforcement Learning')
parser.add_argument('--test', action='store_true', default=False,
                    help='if passed sets to test phase')

args = parser.parse_args()


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def absolute_difference(irs):
    ir_old, ir_new = irs
    sum=0
    for i,val in enumerate(ir_old):
        diff = abs(val - ir_new[i])
        sum+=diff
    return sum

def check_if_collected(rob, i_old):
    i = rob.collected_food()
    if i - i_old ==1:
        collected =True
    else:
        collected = False
    i_old = i
    return collected, i_old


def rob_collectfood(rob, new_state, init_col, action, action_N, episode, t):
    center = 0.5 #center of the image
    #calculate difference between x and y and center
    center_penalty = abs(new_state[3]-center) #[0]
    reward = -1
  #  print(old_state)
    collected, init_col = check_if_collected(rob, init_col)
    if collected: #old_state[2] > 0.6 or
        print('----------------------------------')
        print('Found Food')
        print('-----------------------------------')
        reward=4 #8

        if t>500:
            episode.append(action_N)
            print('episode in the definition', len(episode))
            action_N=0

    elif new_state[3] != 0 and new_state[4] != 0: #0, 1
        if action ==2 and center_penalty<0.1:
            reward = 3
        else:
            reward = 2.5 - center_penalty*4 #2
    else:
        pass
    return reward, action_N, init_col


def camera_input(rob):
    image = rob.get_image_front()

    x_g = 0
    y_g = 0
    size_g = 0
    x_orig =0
    y_orig =0

    if GAUSSIAN:
        x_g = np.random.normal(0,0.01,1)[0]
        y_g = np.random.normal(0,0.01,1)[0]
        size_g = np.random.normal(0,0.01,1)[0]
    else:
        pass

    y_h, x_w, dim = image.shape
    bilFilter = cv2.GaussianBlur(image, (9,9), 2)
    hsv_image = cv2.cvtColor(bilFilter, cv2.COLOR_BGR2HSV)
    # mask_lower = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    # mask_upper = cv2.inRange(hsv_image, (160,100,100), (179,255,255))
    mask_lower = cv2.inRange(hsv_image, (0, 100, 150),(9, 255, 255)) #230 # reduced from 10 to lower becasue 10 is kind of orange
    mask_upper = cv2.inRange(hsv_image, (174, 153, 204), (180, 255, 255))
    mask = cv2.bitwise_or(mask_lower, mask_upper)
    # for green color mask = cv2.inRange(hsv_image, (36, 100, 50), (80, 255, 255))

    # for green color mask = cv2.inRange(hsv_image, (36, 100, 50), (80, 255, 255))

    # implemented treshold
    res = cv2.bitwise_and(image, image, mask=mask)
    # tresholded image
    ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)

    # changed from mask to
    val, contours, hierarchy = cv2.findContours(thrshed, cv2.RETR_EXTERNAL,  # RETR_TREE
                                                cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NON

    try:
        blob = max(contours, key=lambda el: cv2.contourArea(el))
        count_xy, count_wh, angle = cv2.minAreaRect(blob)
        count_w, count_h = count_wh
        size = (count_w / x_w * count_h / y_h) + size_g
        M = cv2.moments(blob)
        try:
            x, y = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x_orig = x
            y_orig = y
            x = (float(x)/x_w ) + x_g
            y = (float(x)/y_h ) + y_g
        except ZeroDivisionError:
            x = 0
            y = 0
            size = 0
    except ValueError:
        x = 0
        y = 0
        size = 0


    timestr = time.strftime("%Y%m%d-%H%M%S")
    canvas = bilFilter.copy()
    cv2.circle(canvas, (x_orig, y_orig), 2, (255, 0, 0), -1)
    cv2.imwrite('red_smooth_real' + timestr + '.png', canvas)
    return x, y, size

def rob_crashed(new_state, reward): #CHANGE THIS 
    count_ir = 0
    additional_reward = 0
    for i,ir_read in enumerate(new_state):
        if ir_read != (-10):
            if ir_read<=(-3): #change treshold to -2.6 (smaller than not equal)
                count_ir+=1
            else:
                additional_reward += abs(ir_read) #CHANGE THIS 
    if count_ir >=2:
        reward=-15 #-6
        crash=True
    else:
        reward += additional_reward
        crash = False
    return reward, crash

def rob_move_food_real(rob,action, old_state):
    if action == 0:
        rob.move(-20, 20, 400) #-10 10 900
    elif action == 1:
        rob.move(20, -20, 400)
    elif action ==2: #go forward
        rob.move(20, 19.6, 1300) #20, 19.6 1000

    x_val, y_val, conture = camera_input(rob)
    new_state = np.asarray([old_state[3], old_state[4], old_state[5], x_val, y_val, conture])
    return new_state

def rob_move_food(rob,action, old_state, ir):
    if action==0: #trun left
        rob.move(-1,1,2000)
    elif action ==1: #turn right
        rob.move(1,-1,2000)
    elif action ==2: #go forward
        rob.move(2,2,3000) #decreased speed 5, 2.5

    x_val, y_val, conture = camera_input(rob)
    new_state=np.asarray([ old_state[3],old_state[4] ,old_state[5],x_val, y_val, conture])
    ir_update = inf_to_null(list(np.log(np.array(rob.read_irs()))/10))
    ir_new = (ir[1],ir_update)
    return new_state, ir_new


def rob_move(rob, action):
    reward = -1
    if action==0: #trun left
        rob.move(-3,3,2000)
    elif action ==1: #turn right
        rob.move(3,-3,2000)
    elif action ==2: #go forward
        rob.move(4,4,8000) #decreased speed 5, 2.5
        reward = 8

    #now the robot ride's for 3 seconds
    new_state = replace_inf(np.log(np.array(rob.read_irs()))/10) 
    return new_state, reward

def rob_move_real(rob, action):
    reward = -1
    if action == 0:
        rob.move(-2, 2, 1000)
    elif action == 1:
        rob.move(4, -4, 1000)
    elif action == 2:
        rob.move(3,2,2000) #left right
        reward = 8

    new_state = replace_inf(rob.read_irs()) #5 values until crash

    return new_state, reward

def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
           str(params['batchSize']) + '-' + str(params['buffer'])

def params_to_filename_prey(params):
    return 'prey'+'-'+str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
           str(params['batchSize']) + '-' + str(params['buffer'])


def process_minibatch(minibatch, model):
    #feed whole batch of data to the model

    mb_len = len(minibatch)

    old_states = np.zeros(shape=(mb_len, NUM_SENSORS))
    actions = np.zeros(shape=(mb_len,))
    rewards = np.zeros(shape=(mb_len,))
    new_states = np.zeros(shape=(mb_len, NUM_SENSORS))

    for i, m in enumerate(minibatch):
        old_state_m, action_m, reward_m, new_state_m = m #has to be a numpy array
        old_states[i, :] = old_state_m[...]
        actions[i] = action_m
        rewards[i] = reward_m
        new_states[i, :] = new_state_m[...]


    old_qvals = model.predict(old_states, batch_size=mb_len)

    new_qvals = model.predict(new_states, batch_size=mb_len)
    maxQs = np.max(new_qvals, axis=1)
    y=old_qvals
    non_term_inds = np.where (rewards != -15)[0]
    y[non_term_inds, actions[non_term_inds].astype(int)] = rewards[non_term_inds] + (GAMMA*maxQs[non_term_inds])
    X_train = old_states
    y_train = y

    return X_train, y_train


def log_results(filename, data_collect, loss_log, dict):

    with open(dict+filename + 'bigSF.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)



def replace_inf(list):
    new_l=np.zeros(len(list))
    for i,val in enumerate(list):
        if str(val) == '-inf':
            new_l[i]=-10
        else:
            new_l[i]=val
    return new_l

def inf_to_null(list):
    new_l = np.zeros(len(list))
    for i, val in enumerate(list):
        if str(val) == '-inf':
            new_l[i] = 0
        else:
            new_l[i] = val
    return new_l


def open_data(filename):
    file = open(filename, 'rb')
    rep = pickle.load(file)
    rep_n=[]
    for r in rep:
        s, a, r1, s1 = r
        s_n = replace_inf(s)
        s1_n = replace_inf(s1)
        rep_n.append([s_n, a, r1, s1_n])
    return rep_n

def check_if_stuck(ir_list, state):
    ir_old, ir_new = ir_list
    difference = 0
    equal = False
    sum_val=0
    if state[2] == 0:
        for i,val in enumerate(ir_new):
            dif = val - ir_old[i]
            difference +=dif
            sum_val += abs(val)
        if difference<0.1 and difference>0 and sum_val>1: #0.01
            equal=True
    else:
        pass
    return equal

def camera_input_prey(rob):
    image = rob.get_image_front()

    #gaussian = False
    x_orig = 0
    y_orig = 0
    x_g = 0
    y_g = 0
    size_g = 0

    if GAUSSIAN:
        x_g = np.random.normal(0, 0.01, 1)[0]
        y_g = np.random.normal(0, 0.01, 1)[0]
        size_g = np.random.normal(0, 0.01, 1)[0]
    else:
        pass

    y_h, x_w, dim = image.shape
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_lower = cv2.inRange(hsv_image, (0, 100, 230),(9, 255, 255))  # reduced from 10 to lower becasue 10 is kind of orange
    mask_upper = cv2.inRange(hsv_image, (174, 153, 204), (180, 255, 255))
    mask = cv2.bitwise_or(mask_lower, mask_upper)


    # implemented treshold
    res = cv2.bitwise_and(image, image, mask=mask)
    # tresholded image
    ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)

    # changed from mask to
    val, contours, hierarchy = cv2.findContours(thrshed, cv2.RETR_EXTERNAL,  # RETR_TREE
                                                cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_NON

    try:
        blob = max(contours, key=lambda el: cv2.contourArea(el))
        count_xy, count_wh, angle = cv2.minAreaRect(blob)
        count_w, count_h = count_wh
        size = (count_w / x_w * count_h / y_h) + size_g
        M = cv2.moments(blob)
        try:
            x_orig, y_orig = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x = (float(x_orig) / x_w) + x_g
            y = (float(x_orig) / y_h) + y_g
        except ZeroDivisionError:
            x = 0
            y = 0
            size = 0
    except ValueError:
        x = 0
        y = 0
        size = 0

    return x, y, size


def _sensor_better_reading(sensors_values):
        """
        Normalising simulation sensor reading due to reuse old code
        :param sensors_values:
        :return:
        """
        old_min = 0
        old_max = 0.20
        new_min = 20000
        new_max = 0
        all = [0 if value is False else (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for value in sensors_values]
        return all[0:3], all[3:8]

def update_state(rob, state):
    x_val, y_val, conture = camera_input_prey(rob)
    new_state=np.asarray([state[3], state[4], state[5], x_val, y_val, conture])
    return new_state

def rob_move_prey(rob,action, old_state):

    if action==0: #trun left
        rob.move(-10,20,200)
    elif action ==1: #turn right
        rob.move(20,-10,200)
    elif action ==2: #go forward
        rob.move(40,40,200) #decreased speed 5, 2.5

    x_val, y_val, conture = camera_input_prey(rob)
    new_state=np.asarray([ old_state[3],old_state[4] ,old_state[5],x_val, y_val, conture])

    ir_back, ir  =  _sensor_better_reading(rob.read_irs())
    return new_state, ir_back, ir

def rob_avoid_pred(new_state, ir_back):
    reward = -1
    close=False
    if new_state[3] !=0 and new_state[4]!= 0 :
        if sum(ir_back)>1000 and sum(ir_back)<12000:
            print('++++++++++++++++++++too close++++++++++++++++++++++++++++')
            reward = -6
            close=True
        else:
            reward_initial = 3
            size_diff = (new_state[2]-new_state[5])*4
            reward = reward_initial + size_diff


    return reward, close

def train_net_prey(model, params, import_data):
    filename = params_to_filename_prey(params)
    gather_observations = 500
    epsilon = 0.9  # aka select actions at random
    train_model = 15000  # number of epsiodes to play
    batchSize = params['batchSize']
    buffer = params['buffer']

    t=0
    #store data
    output_dict = 'src/output_nn_prey/'
    total_reward=0
    total_reward_time=[]
    data_collect=[]
    replay=[] #stores tuples of (s,a,r,s')
    loss_log=[]
    too_close=[]

    if import_data: #set to false if no data present
        t=500
        replay = []
        for i in [100,200, 300, 400, 500]:
            filename1 = output_dict+'replay_F_bigS-'+str(i)+'.pkl'
            rep = open_data(filename1)
            replay.extend(rep)


    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='100.75.3.88', port=19997)
    rob.play_simulation()
    rob.set_phone_pan(9.5, 100)
    rob.set_phone_tilt(101.5, 15)

    time.sleep(1)
    print('simuation started')
    rob.play_simulation()

    #initilaize the preditor in this case
    preditor_robot = robobo.SimulationRoboboPrey().connect(address='100.75.3.88', port=19989)
    preditor_controller = prey.Prey(robot=preditor_robot, level=4)
    preditor_controller._robot.set_phone_tilt(101.5, 15)  # 101.5
    preditor_controller.start()

    #read the inital game state
    x_val, y_val, conture = camera_input_prey(rob)
    state=np.asarray([0,0,0,x_val, y_val, conture])
    #change the ir readings
    ir_back, ir = _sensor_better_reading(rob.read_irs())

    while t<train_model:
        print('Timepoint: ', t)
        t+=1

        if sum(ir) != 0:
            index_max_value = ir.index(max(ir))
            if index_max_value == 2:
                if state[3] < 0.5:  # random.random() <= 0.5:
                    while sum(ir[0:2]) > 500:
                        rob.move(left=-10.0, right=20.0, millis=200)
                        ir_back, ir = _sensor_better_reading(rob.read_irs())
                    state = update_state(rob, state)
                elif state[3] >= 0.5:
                    while sum(ir[3:5]) > 500:  # 6:8
                        rob.move(left=20.0, right=-10, millis=500)
                        ir_back, ir = _sensor_better_reading(rob.read_irs())
                    state = update_state(rob, state)
                elif state[3] == 0:
                    if random.random() <= 0.5:
                        while sum(ir[0:2]) > 500:
                            rob.move(left=-10.0, right=20.0, millis=200)
                            ir_back, ir = _sensor_better_reading(rob.read_irs())
                        state = update_state(rob, state)
                    else:
                        while sum(ir[3:5]) > 500:  # 6:8
                            rob.move(left=20.0, right=-10, millis=500)
                            ir_back, ir = _sensor_better_reading(rob.read_irs())
                        state = update_state(rob, state)
            elif index_max_value == 0 or index_max_value == 1:
                # front right right -> go left
                while sum(ir[0:2]) > 500:  # 3:5
                    rob.move(left=-10.0, right=20.0, millis=200)
                    ir_back, ir= _sensor_better_reading(rob.read_irs())
                state = update_state(rob, state)
            elif index_max_value == 3 or index_max_value == 4:
                # front left left -> go right
                while sum(ir[3:5]) > 500:  # 6:8
                    rob.move(left=20.0, right=-10.0, millis=200)
                    ir_back, ir = _sensor_better_reading(rob.read_irs())
                state=update_state(rob, state)

        #select an action
        if random.random() < epsilon or t<gather_observations:
            action = np.random.randint(0,3) #three actions
            print('Random action selected: ', ACTIONS[action])

        else:
            #get Q val for each action
            state_pred=state.reshape(1,-1)
            qval = model.predict(state_pred,batch_size=1)
            print(qval)
            action = (np.argmax(qval)) #select maximizing action
            print('Greedy action selected: ', ACTIONS[action])

        #take this action, observe the new state and get the corresponding reward
        new_state, ir_back, ir =rob_move_prey(rob, action, state)
        reward, close=rob_avoid_pred(new_state, ir_back)

        if close:
            too_close.append(1)
            print('got too close')
        else:
            too_close.append(0)

        replay.append((state, action, reward, new_state))

        #stop observing start training
        if t > gather_observations:
            print('Starting to train the network')

            #if stored enough in buffer remove the oldest
            if len(replay) > buffer:
                replay.pop(0)

            #sample (number equals the batchSize) from the replay buffer
            minibatch = random.sample(replay, batchSize) #increase the batchSize
            #get the training values
            X_train, y_train = process_minibatch(minibatch, model)

            #train the model on this batch
            history = LossHistory()
            model.fit(X_train, y_train, batch_size=batchSize,
                      nb_epoch=1, verbose=0, callbacks=[history])
            print('Training finished')
            print(history.losses)
            loss_log.append(history.losses)

            #time.sleep(1)

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > gather_observations:
            epsilon -= (1.0 / train_model)

        total_reward += reward
        total_reward_time.append(total_reward)


        if t % 1000 == 0:
            model.save_weights(output_dict+filename+'F_bigS'+'-'+str(t)+'.h5', overwrite=True)
            print('saving model %s - %d'%(filename,t))
            with open(output_dict+filename + 'F_bigS' + str(t)+ '.csv', 'w') as lf:
                wr = csv.writer(lf)
                for loss_item in loss_log:
                    wr.writerow(loss_item)

        if t % 100 == 0:

            with open(output_dict+'total_reward_F_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(total_reward_time, f)

            with open(output_dict+'close_F'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(too_close, f)

            with open(output_dict+'replay_F_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(replay, f)


        state=new_state

    log_results(filename, data_collect, loss_log,output_dict)    

def train_net_video(model, params, import_data):


    filename=params_to_filename(params)
    gather_observations = 500
    epsilon = 0.9 #aka select actions at random
    train_model = 20000 #number of epsiodes to play
    batchSize = params['batchSize']
    buffer = params['buffer']


    t=0
    #store data
    output_dict = 'src/output_files/'
    total_reward=0
    total_reward_time=[]
    data_collect=[]
    replay=[] #stores tuples of (s,a,r,s')
    loss_log=[]
    initial_collection=0

    if import_data: #set to false if no data present
        t=500
        replay = []
        for i in [100,200, 300, 400, 500]: 
            filename1 = output_dict+'replay_F-'+str(i)+'.pkl'
            rep = open_data(filename1)
            replay.extend(rep)

    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)
    rob = robobo.SimulationRobobo().connect(address='145.108.69.180', port=19997)
    rob.play_simulation()
    rob.set_phone_tilt(101.5, 15)
    time.sleep(1)

    print('simuation started')
    rob.play_simulation()

    #read the inital game state
    #state = replace_inf(np.log(np.array(rob.read_irs())))
    x_val, y_val, conture = camera_input(rob)
    state=np.asarray([0,0,0,x_val, y_val, conture])
    ir=inf_to_null(list(np.log(np.array(rob.read_irs()))/10))
    ir_readings=([0,0,0,0,0],ir)
    all_episodes_food=[]
    episode = []
    actions_taken=0

    while t<train_model:
        print('Timepoint: ', t)
        t+=1

        #select an action
        if random.random() < epsilon or t<gather_observations:
            action = np.random.randint(0,3) #three actions
            print('Random action selected: ', ACTIONS[action])

        else:
            #get Q val for each action
            state_pred=state.reshape(1,-1)
            qval = model.predict(state_pred,batch_size=1)
            action = (np.argmax(qval)) #select maximizing action

        # add number of actions
        actions_taken += 1 if t> gather_observations else 0

      #take this action, observe the new state and get the corresponding reward
        new_state, ir_readings = rob_move_food(rob, action, state, ir_readings) #rob,action, old_state, ir):
        reward, actions_taken, initial_collection=rob_collectfood(rob, new_state, initial_collection,action, actions_taken, episode,t)

        #store the environent info
        replay.append((state, action, reward, new_state))

        #stop observing start training
        if t > gather_observations:
            print('Starting to train the network')

            #if stored enough in buffer remove the oldest
            if len(replay) > buffer:
                replay.pop(0)

            #sample (number equals the batchSize) from the replay buffer
            minibatch = random.sample(replay, batchSize) 

            #get the training values
            X_train, y_train = process_minibatch(minibatch, model)

            #train the model on this batch
            history = LossHistory()
            model.fit(X_train, y_train, batch_size=batchSize,
                      nb_epoch=1, verbose=0, callbacks=[history])
            print('Training finished')
            print(history.losses)
            loss_log.append(history.losses)
            time.sleep(1)

        # if all items found or roboo is stuck, re-set the game and continue learning    
        if rob.collected_food()>=6 or check_if_stuck(ir_readings, new_state): #t%1000==0 or
            # print('_______________________stuck_________________________')#t%10==0:#f
            print('ir readings', ir_readings)
            print("----------------------------------------------------------------")
            print('stopped the game')
            all_episodes_food.append(episode)
            rob.stop_world()
            rob.wait_for_stop()
            print('started a new game')
            rob.play_simulation()
            episode = []
            actions_taken=0
            initial_collection = 0


        # Decrement epsilon over time.
        if epsilon > 0.1 and t > gather_observations:
            epsilon -= (1.0 / train_model)

        total_reward += reward
        total_reward_time.append(total_reward)
        time.sleep(1)

        if t % 1000 == 0:
            model.save_weights(output_dict+filename+'F_bigS'+'-'+str(t)+'.h5', overwrite=True)
            print('saving model %s - %d'%(filename,t))
            with open(output_dict+filename + 'F_bigS' + str(t)+ '.csv', 'w') as lf:
                wr = csv.writer(lf)
                for loss_item in loss_log:
                    wr.writerow(loss_item)

            with open(output_dict+'food_steps_F_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(all_episodes_food, f)

        if t % 100 == 0:

            with open(output_dict+'total_reward_F_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(total_reward_time, f)

            with open(output_dict+'replay_F_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(replay, f)

        state=new_state

    log_results(filename, data_collect, loss_log,output_dict)

def train_net_ir(model, params, import_data):
    filename=params_to_filename(params)
    gather_observations = 500
    epsilon = 0.9 #aka select actions at random
    train_model = 20000 #number of epsiodes to play
    batchSize = params['batchSize']
    buffer = params['buffer']

    t=0
    #store data
    output_dict = 'src/output_files/'
    total_reward=0
    total_reward_time=[]
    data_collect=[]
    location=[]
    replay=[] #stores tuples of (s,a,r)
    loss_log=[]
    crash_list=[]

    if import_data: #set to false if no data present
        t=500
        replay = []
        for i in [100,200, 300, 400, 500]: 
            filename1 = output_dict+'replay_F-'+str(i)+'.pkl'
            rep = open_data(filename1)
            replay.extend(rep)

    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)
    rob = robobo.SimulationRobobo().connect(address='145.108.69.180', port=19997)
    rob.play_simulation()
    
    #read the inital game state
    state = replace_inf(np.log(np.array(rob.read_irs())))
    actions_taken=0

    while t<train_model:
        print('Timepoint: ', t)
        t+=1
        location.append(rob.position())

        #select an action
        if random.random() < epsilon or t<gather_observations:
            action = np.random.randint(0,3) #three actions
            print('Random action selected: ', ACTIONS[action])

        else:
            #get Q val for each action
            state_pred=state.reshape(1,-1)
            qval = model.predict(state_pred,batch_size=1)
            action = (np.argmax(qval)) #select maximizing action

        # add number of actions
        actions_taken += 1 if t> gather_observations else 0

      #take this action, observe the new state and get the corresponding reward
        new_state, reward = rob_move(rob, action)
        crash, reward = rob_crashed(new_state, reward)

        #store the environent info
        replay.append((state, action, reward))

        #stop observing start training
        if t > gather_observations:
            print('Starting to train the network')

            #if stored enough in buffer remove the oldest
            if len(replay) > buffer:
                replay.pop(0)

            #sample (number equals the batchSize) from the replay buffer
            minibatch = random.sample(replay, batchSize) 

            #get the training values
            X_train, y_train = process_minibatch(minibatch, model)

            #train the model on this batch
            history = LossHistory()
            model.fit(X_train, y_train, batch_size=batchSize,
                      nb_epoch=1, verbose=0, callbacks=[history])
            print('Training finished')
            print(history.losses)
            loss_log.append(history.losses)
            time.sleep(1)

        #
        if crash:
            # Stopping the simualtion resets the environment
            print('Crashed in an object')
            rob.move(-2,-2,2000)
            print('Moved backward')
            state = replace_inf(np.log(np.array(rob.read_irs())))
            crash_list.append(1)
        else:
            state = new_state
            crash_list.append(0)

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > gather_observations:
            epsilon -= (1.0 / train_model)

        total_reward += reward
        total_reward_time.append(total_reward)
        time.sleep(1)

        if t % 1000 == 0:
            model.save_weights(output_dict+filename+'_bigS'+'-'+str(t)+'.h5', overwrite=True)
            print('saving model %s - %d'%(filename,t))
            with open(output_dict+filename + '_bigS' + str(t)+ '.csv', 'w') as lf:
                wr = csv.writer(lf)
                for loss_item in loss_log:
                    wr.writerow(loss_item)

        if t % 100 == 0:

            with open(output_dict+'total_reward_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(total_reward_time, f)

            with open(output_dict+'location_'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(location, f)

            with open(output_dict+'replay_bigS'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(replay, f)

            with open('crash_correctparam'+'-'+str(t)+'.pkl', 'wb') as f:
                pickle.dump(crash_list, f)

        state=new_state

    log_results(filename, data_collect, loss_log,output_dict)

def play_real_video(filename):
    output_dict = 'src/output_files/'
    saved_model = output_dict + filename
    model = neural_net(NUM_SENSORS, [10, 10], saved_model)

    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)
    print('signal detected')

    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.43.38")
    #tilt the thing
    rob.set_phone_tilt(105, 100, 1)
    time.sleep(1)

    x_val, y_val, conture = camera_input(rob)
    state = np.asarray([0,0,0,x_val, y_val, conture])

    for i in range(10000):
        state_val = state.reshape(1, -1)
        qval = model.predict(state_val, batch_size=1)
        print(qval)
        action = (np.argmax(qval))  # select maximizing action


        new_state = rob_move_food_real(rob, action, state)
        print('New state', new_state)

        state = new_state

def play_sim_video(filename):
    output_dict = 'src/output_files/'
    saved_model = output_dict + filename #'10-10-64-1000F_bigS-15000.h5'

    model = neural_net(NUM_SENSORS, [10, 10], saved_model)
    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='145.108.70.95', port=19997)
    rob.set_phone_tilt(101.5, 15)
    time.sleep(1)
    print('signal created')

    #start the simulation
    rob.play_simulation()

    # read the inital game state
    x_val, y_val, conture = camera_input(rob)
    state = np.asarray([0,0,0,x_val, y_val, conture])
    ir = inf_to_null(list(np.log(np.array(rob.read_irs())) / 10))
    ir_readings = ([0, 0, 0, 0, 0], ir)

    for i in range(10000):
        state_val = state.reshape(1, -1)
        qval = model.predict(state_val, batch_size=1)
        action = (np.argmax(qval))  # select maximizing action
        new_state, ir_readings = rob_move_food(rob, action,state, ir_readings)
        state = new_state

def play_sim_ir(filename):
    output_dict = 'src/output_files/'
    saved_model = output_dict + filename

    model = neural_net(NUM_SENSORS, [10, 10], saved_model)
    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='145.108.70.95', port=19997)
  #  rob.set_phone_tilt(101.5, 15)
    time.sleep(1)
    print('signal created')

    #start the simulation
    rob.play_simulation()

    # read the inital game state
    state = replace_inf(np.log(np.array(rob.read_irs()))/10) #for obstacle

    for i in range(10000):
        state_val = state.reshape(1, -1)
        qval = model.predict(state_val, batch_size=1)
        action = (np.argmax(qval))  # select maximizing action

        new_state, _ = rob_move(rob, action)
        state = new_state

def rotationsimul():
     #start new game instance
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.1.7")
    #rob = robobo.SimulationRobobo().connect(address='145.108.69.97', port=19997)
    print('signal detected')

    rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.5")
    #172.20.10.5 = ip of robobo
    #172.20.10 = ROSMASTER SET  
    #145.108.68.115 = ip computer
    #tilt the thing
    print('till here')
    rob.set_phone_tilt(105, 100, 1)

    time.sleep(1)

    #start the simulation
    #rob.play_simulation()
    #rob.talk('Hi humans, my name is Robobaby')
    #read the inital game state
    #rob.set_phone_tilt(103, 20)
    for i in range(20):
        print(i)
        rob_move(rob, 2)

def sensor_better_reading(sensors_values): #self,
    """
    Normalising simulation sensor reading due to reuse old code
    :param sensors_values:
    :return:
    """
    old_min = 0
    old_max = 0.20
    new_min = 20000
    new_max = 0
    return [0 if value == '-inf' else (((value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min for value in sensors_values]

def play_prey(level):
    signal.signal(signal.SIGINT, terminate_program)
    print('signal detected')

    rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.5")

    rob.set_phone_tilt(105, 100, 1)
    time.sleep(1)

    if level == 0:
        print("Level 0 selected -> super easy option")
        maximum_speed = 5.0
        turning_speed = 5.0
        epsilon = 0.02
    elif level == 1:
        print("Level 1 selected -> easy option")
        maximum_speed = 10.0
        turning_speed = 10.0
        epsilon = 0.03
    elif level == 2:
        print("Level 2 selected -> medium option")
        maximum_speed = 20.0
        turning_speed = 10.0
        epsilon = 0.06
    elif level == 3:
        print("Level 3 selected -> hard option".format(self._level))
        maximum_speed = 40.0
        turning_speed = 20.0
        epsilon = 0.08
    elif level == 4:
        print("Level 4 selected -> insane (Good luck Catching Me)".format(self._level))
        maximum_speed = 70.0
        turning_speed = 30.0
        epsilon = 0.1
    else:
        raise Exception("Level Value not correct, try from 0 to 4.")

    while True:
        speed_right = speed_left = maximum_speed
        if random.random() <= epsilon:
            speed_right = random.uniform(-turning_speed, turning_speed)
            speed_left = random.uniform(-turning_speed, turning_speed)
            for i in range(3):
                print(i)
                rob.move(speed_right, speed_left, 200)
        rob.move(speed_right, speed_left, 200)
        sensors = sensor_better_reading(rob.read_irs())
        if sum(sensors) != 0:
            index_max_value = sensors.index(max(sensors))
            if index_max_value == 5:
                # central -> turn
                if random.random() <= 0.5:
                    while sum(sensors[3:5]) > 500:
                        rob.move(10.0, 20.0, 500)
                        sensors = sensor_better_reading(rob.read_irs())
                else:
                    while sum(sensors[6:8]) > 500:
                        rob.move(20.0, -10, 500)
                        sensors = sensor_better_reading(rob.read_irs())
            elif index_max_value == 3 or index_max_value == 4:
                while sum(sensors[3:5]) > 500:
                    rob.move(-10.0, 20.0, 200)
                    sensors = sensor_better_reading(rob.read_irs())
                    print("front right right -> go left")
                    print(sensors)
            elif index_max_value == 7 or index_max_value == 6:
                while sum(sensors[6:8]) > 500:
                    rob.move(20.0, 0.0, 200)
                    sensors = sensor_better_reading(rob.read_irs())

def play_intelligent_prey(filename):
    output_dict = 'src/output_files/'
    saved_model = output_dict + filename
    model = neural_net(NUM_SENSORS, [10, 10], saved_model)

    #start new game instance
    signal.signal(signal.SIGINT, terminate_program)
    print('signal detected')

    rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.5")
    #tilt the thing
    rob.set_phone_tilt(105, 100, 1)
    time.sleep(1)
    x_val, y_val, conture = camera_input(rob)
    state = np.asarray([x_val, y_val, conture])

    for i in range(10000):
        state_val = state.reshape(1, -1)
        qval = model.predict(state_val, batch_size=1)
        action = (np.argmax(qval))  # select maximizing action
        new_state=rob_move_prey(rob,action,state_val)
        print('New state', new_state)

        state = new_state

def check_wall(rob, count_ir):
    sensors = rob.read_irs()
    for value in sensors:
        if value>4: 
            count_ir+=1 
    if count_ir >=3:
        no_wall='false'
    else:
        no_wall ='true'
    return no_wall

def play_prey_own(level):
    signal.signal(signal.SIGINT, terminate_program)
    print('signal detected')

    rob = robobo.HardwareRobobo(camera=True).connect(address="172.20.10.5")
    rob.set_phone_pan(343,100, pan_blockid=1)
    time.sleep(1)
    rob.set_phone_tilt(105, 100, 1)
    time.sleep(1)

    # rob = robobo.SimulationRobobo().connect(address='172.20.10.3', port=19997)
    # rob.play_simulation()
    # set camera position simulation
    # rob.set_phone_pan(500, 100)
    # rob.set_phone_tilt(101.5, 15)
    # time.sleep(1)
    # print('simuation started')
    
    #option 2 focused on forward + random
    if level==1:
        for k in range(1000):
            if k % 5 == 0:
                rob_move_real(rob, 1)
            elif k % 18 == 0:
                rob_move_real(rob, 0)
            else:
                rob_move_real(rob, 2)
    
    #option 3 avoiding wall
    if level ==2:
        for i in range(1000):
            count_ir = 0
            no_wall = check_wall(rob, count_ir)
            if no_wall == 'true':
                rob_move_real(rob, 2)
            else:
                rob_move_real(rob, 0)


if __name__ == "__main__":
    if not args.test:
        #run in simuator
        import_data = False #set to true if have previously trained simulations
        # NN parameter settings 
        nn_param=[10,10] #number of weights (one hidden state)
        params = {
        "batchSize":64,
        "buffer":1000,
        "nn":nn_param
        }

        #initialize model 
        model = neural_net(NUM_SENSORS, nn_param)

        print('model initialized')

        #experiment 1: obstacle avoidance
        train_net_ir(model, params, import_data)
        
        #experiment 2: foraging task
        train_net_video(model, params, import_data) 

        #experiment 3: predator chasing a prey (re-used trained predator model)
        train_net_prey(model,params, import_data) 

    else:
        #define which model to load
        filename = '0-10-64-1000F_bigS-15000.h5'

        # run experiment 1 in simulation with trained model 
        play_sim_ir(filename)

        #run experiment 2 or 3 in simulation with the trained model
        play_sim_video(filename)

        #run experiment 2 or 3 on hardware with trained model 
        play_real_video(filename)
        
        # if playing as prey on hardware
        level = 0
        play_prey_own(level)
        play_intelligent_prey(filename) 






   



