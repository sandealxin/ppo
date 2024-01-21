import airsim
from agent import PPO

import airsim
import time
import numpy as np
import threading
import json
import os
import uuid
import glob
import datetime
import h5py
import sys
import requests
import PIL
import copy
import datetime
import math
import tensorflow as tf
import random
import multiprocessing
import queue

min_speed = 0.1

# A class that represents the agent that will drive the vehicle, train the model, and send the gradient updates to the trainer.
class DistributedAgent():
    def __init__(self,agents, __multirotor_client):        
        print('Starting time: {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)
        self.agents = agents
        self.n_agents = len(self.agents)
        self.vehicle_name = None
    
        self.__model = PPO(4, [[144,256,3],3])
        #self.__model.temp_memory[self.vehicle_name] = []
        
        # Agent attribute
        self.agent_old_location = [None]*self.n_agents
        self.agent_current_location = [None]*self.n_agents

        # Setup airsim client
        self.__multirotor_client = __multirotor_client
        self.pose = self.__multirotor_client.simGetVehiclePose()
        
        self.__init_road_points()
        self.__init_reward_points()
        
        #Init target list
        self.__target_point_list = []
        self.__init_target_point_list()
        self.__target_point = random.choice(self.__target_point_list).astype(np.float64)
        
        self.max_dist_to_target = 99999
        
    # Starts the agent
    def start(self):
        self.__run_function()

    # The function that will be run during training.
    # It will initialize the connection to the trainer, start AirSim, and continuously run training iterations.
    def __run_function(self):
        while True:
            self.__run_airsim_epoch()

    # Runs an interation of data generation from AirSim.
    # Data will be saved in the buffer memory.
    def __run_airsim_epoch(self):
        # Pick a random starting point on the roads
        starting_points = self._get_new_start_locations(self.n_agents)
        temp = [None]*self.n_agents
        # Initialize the state buffer.
        # For now, save 4 images at 0.01 second intervals.
        wait_delta_sec = 0.01
        for i in range(self.n_agents):
            self.pose.position.x_val = starting_points[i][0]
            self.pose.position.y_val = starting_points[i][1]
            self.pose.position.z_val = -2
            self.__multirotor_client.simSetVehiclePose(self.pose, True, vehicle_name = self.agents[i])
            #Start the multirotor rolling so it doesn't get stuck
            temp[i] = self.__multirotor_client.moveByRollPitchYawThrottleAsync(0,0,0,1,2,vehicle_name = self.agents[i])
            
        for i in range(self.n_agents):
            temp[i].join()       
        
        #done = [False for i in range(self.n_agents)]
        action = [None]*self.n_agents
        prob = [None]*self.n_agents
        val = [None]*self.n_agents
        observation = [None]*self.n_agents
        multirotor_old_position = [None]*self.n_agents
        
        for i in range(self.n_agents):
            # Get old state
            state = self.__get_image(self.agents[i])
            # Save agent's old location
            self.agent_old_location[i] = self.__get_multirotor_location(self.agents[i])
            observation[i] = [state, self.agent_old_location[i]]
        far_off = False
        i=0
        # Main data collection loop
        while True:
            
            for i in range(self.n_agents):
                
                # Convert the selected state to a control signal
                action[i] , prob[i], val[i] = self.__model.choose_action(copy.deepcopy(observation[i]))
                
                #Squash value in range [-45,45] degree
                roll_val = (tf.math.sigmoid(action[i][0].item())* 90 - 45).numpy().item()
                pitch_val = (tf.math.sigmoid(action[i][1].item())* 90 - 45).numpy().item()
                yaw_val = (tf.math.sigmoid(action[i][2].item())* 90 - 45).numpy().item()
                #Squash value in range (0,1)
                throttle_val = (tf.math.sigmoid(action[i][3].item())).numpy().item()
                # Take the action
                print([roll_val,pitch_val,yaw_val,throttle_val])
                temp[i] = self.__multirotor_client.moveByRollPitchYawThrottleAsync(roll_val,pitch_val,yaw_val,throttle_val,1,vehicle_name = self.agents[i]).join()
            '''
            for i in range(self.n_agents):
                temp[i].join()
            '''

            for i in range(self.n_agents):
                # Save agent's new location
                self.agent_current_location[i] = self.__get_multirotor_location(self.agents[i])

                # Observe outcome and compute reward from action
                multirotor_state = self.__multirotor_client.getMultirotorState(self.agents[i])
                reward, done = self.__compute_reward(multirotor_state,self.agent_old_location[i],self.agent_current_location[i],self.agents[i])

                # Add the experience to the set of examples from this iteration
                self.__model.store_transition(observation[i][0], observation[i][1], action[i], prob[i], val[i], reward, done)

                # Update observation
                state = self.__get_image(self.agents[i])
                
                self.agent_old_location[i] = self.agent_current_location[i]
                observation[i] = [state,  self.agent_old_location[i]]   

                if done:
                    starting_points = self._get_new_start_locations(1)
                    self.pose.position.x_val = starting_points[0][0]
                    self.pose.position.y_val = starting_points[0][1]
                    self.pose.position.z_val = -2
                    self.__multirotor_client.simSetVehiclePose(self.pose, True, vehicle_name = self.agents[i])
            if len(self.__model.memory.states) % 16 == 0:
                for x in range(i):
                    self.__model.learn()
                i+=1
            if i == 10:
                self.__multirotor_client.wait_key('Press any key to continue')
                i = 0

    # Gets an image from AirSim
    def __get_image(self,vehicle_name):
        image_response = self.__multirotor_client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)],vehicle_name = vehicle_name)[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 3)

        return image_rgba[:,:,0:3].astype(float)
    
    def __get_multirotor_location(self,vehicle_name):
        position = self.__multirotor_client.simGetVehiclePose(vehicle_name = vehicle_name).position
        return np.array([position.x_val, position.y_val, position.z_val]).astype(np.float64)
    
    def __scale_outputs(self, outputs, min_bound, max_bound):
    # Assuming outputs are in the range [-1, 1]
    # Scale them to the range [min_bound, max_bound]
        return min_bound + (max_bound - min_bound) * (outputs + 1) / 2


    # Computes the reward functinon based on the multirotor position.
    def __compute_reward(self, multirotor_state,agent_old_location,agent_current_location,vehicle_name):
        collision_info = self.__multirotor_client.simGetCollisionInfo(vehicle_name)
        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=10)
        utc_now = datetime.datetime.utcnow()
        start_time_0 = time.time()
        # Dictionary mapping conditions to their messages
        condition_messages = {
                collision_info.has_collided: "Collision detected.",
                self.__multirotor_client.getMultirotorState(vehicle_name = vehicle_name).kinematics_estimated.linear_velocity.get_length() < min_speed: f"Speed below minimum threshold of {min_speed}.",
                utc_now > end_time: "Time limit exceeded.",
                #far_off: "Far off condition met."
            }
            
        # Check the compound condition
        if any(condition_messages.keys()):
            # Find which condition(s) is/are true and print the corresponding message(s)
            for condition, message in condition_messages.items():
                if condition:
                    print(message, file=sys.stderr)
                    done = True
            reward = -100
            return reward, done

        # Process the state for this agent (since no collision occurred)
        orientation = multirotor_state.kinematics_estimated.orientation
        old_distance = np.linalg.norm(agent_old_location - self.__target_point)
        new_distance = np.linalg.norm(agent_current_location - self.__target_point)

        # Initialize reward and status
        reward = 5 if new_distance < old_distance else -1
        reward += 1 if orientation.x_val < 0.1 or orientation.y_val < 0.1 else 0
        reward = 100 if new_distance < 1 else reward
        done = new_distance < 1

        return reward, done


    # Initializes the target point used for determining the target point of the vehicle
    def __init_target_point_list(self):
        with open('target_points.txt', 'r') as f:
            for line in f:
                target_location = np.array(line.split(','))
                self.__target_point_list.append(target_location)
                
    # Initializes the points used for determining the starting point of the vehicle
    def __init_road_points(self):
        self.__road_points = []
        multirotor_start_coords = [12961.722656, 6660.329102, 0]

        with open('road_lines.txt', 'r') as f:
            for line in f:
                points = line.split('\t')
                first_point = np.array([float(p) for p in points[0].split(',')] + [0])
                second_point = np.array([float(p) for p in points[1].split(',')] + [0])
                self.__road_points.append(tuple((first_point, second_point)))

        # Points in road_points.txt are in unreal coordinates
        # But multirotor start coordinates are not the same as unreal coordinates
        for point_pair in self.__road_points:
            for point in point_pair:
                point[0] -= multirotor_start_coords[0]
                point[1] -= multirotor_start_coords[1]
                point[0] /= 100
                point[1] /= 100
              
    # Initializes the points used for determining the optimal position of the vehicle during the reward function
    def __init_reward_points(self):
        self.__reward_points = []
        with open('reward_points.txt', 'r') as f:
            for line in f:
                point_values = line.split('\t')
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
                self.__reward_points.append(tuple((first_point, second_point)))

    # Randomly selects a starting point on the road
    # Used for initializing an iteration of data generation from AirSim

    def _get_new_start_locations(self,n_agents):
        return np.random.uniform(150, 200, size=(n_agents, 3))
    
    # A helper function to make a directory if it does not exist
    def __make_dir_if_not_exist(self, directory):
        if not (os.path.exists(directory)):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
 
if __name__ == '__main__':
    print('------------STARTING AGENT----------------')
    # Connect to airsim
    __multirotor_client = airsim.MultirotorClient()
    __multirotor_client.confirmConnection()
    exist_drone_name = __multirotor_client.listVehicles()
    
    agents = ['Drone']
    n_agents = 2
    for i in range(n_agents):
        agent_name = "Drone" + str(i)
        agents.append(agent_name)
        pose = airsim.Pose(airsim.Vector3r(i, i, 0), airsim.to_quaternion(0, 0, 0))
        if agent_name not in exist_drone_name:
           __multirotor_client.simAddVehicle(agent_name, "simpleflight", pose)

    for agent in agents:
        __multirotor_client.enableApiControl(True, agent)
        __multirotor_client.armDisarm(True, agent)

    agent = DistributedAgent(agents, __multirotor_client)
    agent.start()
    

        


