import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, Input, Add
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from time import time
import numpy as np
import os
import random
sess = tf.Session()
graph = tf.get_default_graph()

class Network():
    def __init__(self, input_shape, policy_output_shape, learning_rate, momentum, l2):
        self.model = None
        self.learning_rate = learning_rate
        self.momentum = learning_rate
        self.input_shape = input_shape
        self.policy_output_shape = policy_output_shape
        self.l2 = l2
        self.predict_times = 0
        self.predict_average_time = 0.0
        self.predict_total_time = 0
    
    def create_network(self, residual_blocks):
        inputs = Input(shape=self.input_shape)
        x = inputs
        x = Conv2D(64, kernel_size=(3,3), padding='same', data_format='channels_first', kernel_regularizer=l2(self.l2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        for _ in range(residual_blocks):
          x = self.get_residual_block(x)
        network_output = x
        policy_head_output = self.get_policy_head(network_output, self.policy_output_shape)
        value_head_output = self.get_value_head(network_output)
        self.model = Model(inputs=inputs, outputs=[policy_head_output, value_head_output])

         # Define the Loss Function (stochastic gradient descend)
        opt = SGD(lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        losses_type = ['categorical_crossentropy', 'mean_squared_error'] 
        self.model.compile(optimizer=opt, loss=losses_type)

    def get_residual_block(self, block_input):
        x = Conv2D(64, kernel_size=(3,3), padding='same', data_format='channels_first', kernel_regularizer=l2(self.l2))(block_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(3,3), padding='same', data_format='channels_first', kernel_regularizer=l2(self.l2))(block_input)
        x = BatchNormalization()(x)
        x = Add()([x, block_input])
        block_output = Activation('relu')(x)
        return block_output
    
    def get_policy_head(self, policy_input, output_shape):
        x = Conv2D(2, kernel_size=(1,1), padding='same', data_format='channels_first', kernel_regularizer=l2(self.l2))(policy_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(output_shape, kernel_regularizer=l2(self.l2))(x)
        policy_output = Activation('softmax')(x)
        return policy_output

    def get_value_head(self, value_input):
        x = Conv2D(1, kernel_size=(1,1), padding='same', data_format='channels_first', kernel_regularizer=l2(self.l2))(value_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(64, kernel_regularizer=l2(self.l2))(x)
        x = Activation('relu')(x)
        x = Dense(1, kernel_regularizer=l2(self.l2))(x)
        policy_output = Activation('tanh')(x)
        return policy_output

    def predict(self, state, color):
        self.predict_times += 1
        start = time()  
        (state_aug, player_aug) = self.augment_state(state, color) 
        input_data = np.array([self.create_input_tensor(state_aug[i], player_aug[i]) for i in range(len(state_aug))])
        with graph.as_default():
            set_session(sess)
            (p, v) = self.model.predict(input_data)
        self.predict_total_time += time() - start
        self.predict_average_time = self.predict_total_time / self.predict_times
        pred_index = random.choice([i for i in range(len(input_data))])
        return (p[pred_index], v[pred_index][0])

    def create_input_tensor(self, state, current_player):
        
        current_player_layer = np.array(state == current_player, dtype=np.int)
        opponent_player_layer = np.array(state == -current_player, dtype=np.int)

        color_to_play_layer = np.zeros((state.shape[0], state.shape[1]))
        if current_player == 1:
            color_to_play_layer += 1

        input_tensor = np.array([current_player_layer, opponent_player_layer, color_to_play_layer])

        return input_tensor
   
    def save_model(self, path):
        with graph.as_default():
            set_session(sess)
            self.model.save(path)

    def load_model(self, path):
        if os.path.exists(path):
            with graph.as_default():
                set_session(sess)
                self.model = load_model(path)
        else:
            print(f'Error: unable to load model file "{path}"!')

    def train_model(self, state_list, pi_list, z_list, player_list):

      (state_aug, pi_aug, z_aug, player_aug) = self.augment_data(state_list, pi_list, z_list, player_list)

      train_x = np.array([self.create_input_tensor(state_aug[i], player_aug[i]) for i in range(len(state_aug))])
      train_y = np.array(pi_aug)
      train_z = np.array(z_aug)

      self.model.fit(train_x, [train_y, train_z], epochs=20, batch_size = 512)

    def augment_state(self, state, player):
        state_aug = []        
        player_aug = []

        state_aug.append(state)
        state_aug.append(np.flip(state, 1))       
        player_aug.append(player)
        player_aug.append(player)
        return (state_aug, player_aug)

    def augment_data(self, state_list, pi_list, z_list, player_list):
        state_aug = []
        pi_aug = []
        z_aug = []
        player_aug = []

        for i in range(len(state_list)):
            state_aug.append(state_list[i])
            state_aug.append(np.flip(state_list[i], 1))
            pi_aug.append(pi_list[i])
            pi_aug.append(list(np.flip(pi_list[i], 0)))
            z_aug.append(z_list[i])
            z_aug.append(z_list[i])
            player_aug.append(player_list[i])
            player_aug.append(player_list[i])
        return (state_aug, pi_aug, z_aug, player_aug)