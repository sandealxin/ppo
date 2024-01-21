import time
import numpy as np
import json
import threading
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ELU
from tensorflow.keras.optimizers import Adam, SGD, Adamax, Nadam, Adagrad, Adadelta
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import tensorflow.keras as keras

# Use TensorFlow's Keras backend
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.initializers import random_normal
import tensorflow_probability as tfp


from memory import PPOMemory

# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch, as it will be unable to 


# A wrapper class for the DQN model
class PPO:
    def __init__(self, action_dim, state_dim, gamma=0.99,
                 alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=128,
                 n_epochs=1, chkpt_dir='models/'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = gamma
        
        self.alpha = alpha
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size

        self.n_epochs = n_epochs
        self.chkpt_dir = chkpt_dir

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.memory = PPOMemory(128)
        self.memory.load_from_file()
        self.load_models()
        '''
        print('Loading weights from my_model_weights.h5...')
        print('Current working dir is {0}'.format(os.getcwd()))
        self.actor.load_weights('pretrain_model_weights.h5', by_name=True)
        self.critic.load_weights('pretrain_model_weights.h5', by_name=True)

        if (weights_path is not None and len(weights_path) > 0):
            print('Loading weights from my_model_weights.h5...')
            print('Current working dir is {0}'.format(os.getcwd()))
            self.actor.load_weights('pretrain_model_weights.h5', by_name=True)
            self.critic.load_weights('pretrain_model_weights.h5', by_name=True)
        else:
            print('Not loading weights')
        '''



    def store_transition(self, state, position, action, prob, val, reward, done):
        self.memory.store_memory(state, position, action, prob, val, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor')
        self.critic.save(self.chkpt_dir + 'critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic')


    def build_actor(self):
        activation = 'relu'
        #Create the convolutional stacks
       
        pic_input = Input(shape=(self.state_dim[0]))

        img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
        img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
        img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Flatten()(img_stack)
        img_stack = Dropout(0.2)(img_stack)

        #Inject the state input
        state_input = Input(shape=(self.state_dim[1]))
        merged = concatenate([img_stack, state_input])

        # Add a few dense layers to finish the model
        merged = Dense(64, activation=activation, name='merge0')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(10, activation=activation, name='merge1')(merged)
        merged = Dropout(0.2)(merged)

        mean = Dense(self.action_dim, activation='tanh')(merged)
        std = Dense(self.action_dim, activation='softplus')(merged)
        #action = Lambda(lambda x: x[0] +tf.random.normal(tf.shape(x[0])) * x[1])([mean, std])
                            
        actor = Model(inputs=[pic_input, state_input], outputs=[mean,std])
        actor.compile(optimizer=Adam(lr=self.alpha))
        return actor


    def build_critic(self):
        activation = 'relu'
        pic_input = Input(shape=(self.state_dim[0]))

        img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
        img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
        img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
        img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
        img_stack = Flatten()(img_stack)
        img_stack = Dropout(0.2)(img_stack)

        #Inject the state input
        state_input = Input(shape=(self.state_dim[1]))
        merged = concatenate([img_stack, state_input])

        # Add a few dense layers to finish the model
        merged = Dense(64, activation=activation, name='merge0')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(64, activation=activation, name='merge1')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(64, name='output0')(merged)
                            
        value = Dense(1, activation='linear')(merged)
        critic = Model(inputs=[pic_input, state_input], outputs=value)
        critic.compile(optimizer=Adam(lr=self.alpha))
        return critic


    def choose_action(self, observation):
        observation[0] = np.expand_dims(observation[0], axis=0)
        observation[1] = np.expand_dims(observation[1], axis=0)
        
        image = tf.convert_to_tensor(observation[0], dtype=tf.float32)
        position = tf.convert_to_tensor(observation[1], dtype=tf.float32)
        state = [image, position]
        
        mean, std = self.actor(state)
        dist = tfp.distributions.Normal(mean, std)
        #action = tf.clip_by_value(dist.sample(), self.action_bound[0], self.action_bound[1])
        action = dist.sample()

        log_prob = tf.reduce_sum(dist.log_prob(action))
        
        value = self.critic(state)

        action = action.numpy()[0]
        log_prob = log_prob.numpy()
        value = value.numpy()[0]

        return action, log_prob, value

    def learn(self):
        print("-------------------------------------------Start learning ")
        print(len(self.memory.dones))
        for _ in range(self.n_epochs):
            state_arr, position_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (
                        1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                if max(batch) >= len(position_arr):
                    print(f"Invalid batch index detected: {max(batch)}")
                    continue

                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    positions = tf.convert_to_tensor(position_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    observartion = [states, positions]

                    mean, std = self.actor(observartion)
                    dist = tfp.distributions.Normal(mean[0], std[0])
                    new_probs = tf.reduce_sum(dist.log_prob(actions))

                    critic_value = self.critic(observartion)

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

        print("---------------------------Finish learning, save memory ")
        #self.memory.clear_memory()
        print("saving model ")
        self.save_models()
        self.memory.save_to_file()

