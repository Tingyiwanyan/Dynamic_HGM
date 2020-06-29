import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

class LSTM_model():
    """
    Create LSTM model for EHR data
    """
    def __init__(self,kg,data_process):
        """
        initialization for varies variables
        """
        self.kg = kg
        self.data_process = data_process
        #self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.length_train_hadm = len(data_process.train_hadm_id)
        self.batch_size = 16
        self.time_sequence = 4
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.epoch = 1
        self.item_size = len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.input_seq = []
        self.threshold = 0.5
        """
        define LSTM variables
        """
        self.init_hiddenstate = tf.placeholder(tf.float32, [None, self.latent_dim])
        self.input_y_logit = tf.placeholder(tf.float32, [None, 2])
        self.input_x = tf.placeholder(tf.float32,[None,self.time_sequence,self.item_size])
        self.input_y_diag_single = tf.placeholder(tf.float32,[None,self.diagnosis_size])
        self.input_y_diag = tf.placeholder(tf.float32,[None,self.time_sequence,self.diagnosis_size])
        self.init_forget_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state = tf.keras.initializers.he_normal(seed=None)
        self.init_output_gate = tf.keras.initializers.he_normal(seed=None)
        self.init_softmax_convert = tf.keras.initializers.he_normal(seed=None)
        self.init_forget_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_info_gate_weight = tf.keras.initializers.he_normal(seed=None)
        self.init_cell_state_weight = tf.keras.initializers.he_normal(seed=None)
        self.weight_forget_gate = \
            tf.Variable(self.init_forget_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_info_gate = \
            tf.Variable(self.init_info_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_cell_state = \
            tf.Variable(self.init_cell_state(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.weight_softmax_convert = \
            tf.Variable(self.init_softmax_convert(shape=(self.latent_dim,self.diagnosis_size)))
        self.weight_output_gate = \
            tf.Variable(self.init_output_gate(shape=(self.item_size+self.latent_dim,self.latent_dim)))
        self.bias_forget_gate = tf.Variable(self.init_forget_gate_weight(shape=(self.latent_dim,)))
        self.bias_info_gate = tf.Variable(self.init_info_gate_weight(shape=(self.latent_dim,)))
        self.bias_cell_state = tf.Variable(self.init_cell_state_weight(shape=(self.latent_dim,)))
        self.bias_softmax_convert = tf.Variable(self.init_softmax_convert(shape=(self.diagnosis_size,)))
        self.bias_output_gate = tf.Variable(self.init_output_gate(shape=(self.latent_dim,)))



    def lstm_cell(self):
        cell_state = []
        hidden_rep = []
        for i in range(self.time_sequence):
            x_input_cur = tf.gather(self.input_x, i, axis=1)
            if i == 0:
                concat_cur = tf.concat([self.init_hiddenstate,x_input_cur],1)
            else:
                concat_cur = tf.concat([hidden_rep[i-1],x_input_cur],1)
            forget_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_forget_gate),self.bias_forget_gate))
            info_cur = \
                tf.math.sigmoid(tf.math.add(tf.matmul(concat_cur,self.weight_info_gate),self.bias_info_gate))
            cellstate_cur = \
                tf.math.tanh(tf.math.add(tf.matmul(concat_cur,self.weight_cell_state),self.bias_cell_state))
            info_cell_state = tf.multiply(info_cur, cellstate_cur)
            if not i ==0:
                forget_cell_state = tf.multiply(forget_cur, cell_state[i - 1])
                cellstate_cur = tf.math.add(forget_cell_state,info_cell_state)
            output_gate = \
                tf.nn.relu(tf.math.add(tf.matmul(concat_cur,self.weight_output_gate),self.bias_output_gate))
            hidden_current = tf.multiply(output_gate,cellstate_cur)
            cell_state.append(cellstate_cur)
            hidden_rep.append(hidden_current)

        self.hidden_last = hidden_rep[self.time_sequence-1]
        for i in range(self.time_sequence):
            hidden_rep[i] = tf.expand_dims(hidden_rep[i],1)
        self.hidden_rep = tf.concat(hidden_rep,1)
        self.check = concat_cur



    def softmax_loss(self):
        """
        Implement softmax loss layer
        """
        self.output_layer = tf.layers.dense(inputs=self.hidden_last,
                                           units=2,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        #self.logit_sig = tf.math.sigmoid(self.output_layer)
        self.logit_sig = tf.nn.softmax(self.output_layer)
        #self.cross_entropy = tf.reduce_mean(tf.math.negative(
        #    tf.reduce_sum(tf.math.multiply(self.input_y_diag_single, tf.log(self.logit_softmax)), reduction_indices=[1])))
        """
        self.cross_entropy = \
            tf.reduce_mean(
            tf.math.negative(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.math.multiply(
                            self.input_y_diag,tf.log(
                                self.logit_softmax)),reduction_indices=[1]),reduction_indices=[1])))
        """
        """
        self.cross_entropy = tf.reduce_mean(tf.math.negative(
            tf.reduce_sum(tf.math.multiply(self.input_y_logit, tf.log(self.logit_sig)), axis=1)),
            axis=0)
        """
        self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)


    def config_model(self):
        """
        Model configuration
        """
        self.lstm_cell()
        self.softmax_loss()
        self.train_step_cross_entropy = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def assign_value_patient(self,patientid,time):
        one_sample = np.zeros(self.item_size)
        for i in self.kg.dic_patient[patientid]['prior_time'][str(time)].keys():
            mean = self.kg.dic_item[i]['mean_value']
            std = self.kg.dic_item[i]['std']
            ave_value = np.mean(self.kg.dic_patient[patientid]['prior_time'][str(time)][i])
            index = self.kg.dic_item[i]['item_index']
            if std == 0:
                one_sample[index] = 0
            else:
                one_sample[index] = (np.float(ave_value) - mean) / std

        return one_sample

    def get_batch_train(self,data_length,start_index,data):
        """
        get training batch data
        """
        train_one_batch = np.zeros((data_length,self.time_sequence,self.item_size))
        one_batch_logit = np.zeros((data_length,2))
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            for j in self.time_seq_int:
                if time_index == self.time_sequence:
                    break
                #self.time_index = np.int(j)
                self.one_data = self.assign_value_patient(self.patient_id,j)
                train_one_batch[i,time_index,:] = self.one_data
                flag = self.kg.dic_patient[self.patient_id]['flag']
                if flag == 0:
                    one_batch_logit[i, 0] = 1
                else:
                    one_batch_logit[i, 1] = 1
                time_index += 1

        return train_one_batch, one_batch_logit


    def train(self):
        """
        train the system
        """
        init_hidden_state = np.zeros((self.batch_size,self.latent_dim))
        iteration = np.int(np.floor(np.float(self.length_train)/self.batch_size))
        for j in range(self.epoch):
            print('epoch')
            print(j)
            for i in range(iteration):
                train_one_batch, self.logit_one_batch = self.get_batch_train(self.batch_size,i*self.batch_size,self.train_data)
                self.err_ = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_sig],
                                     feed_dict={self.input_x: train_one_batch,
                                                self.input_y_logit: self.logit_one_batch,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_[0])

    def test(self):
        """
        test the system, return the accuracy of the model
        """
        init_hidden_state = np.zeros((self.length_test, self.latent_dim))
        self.test_data, self.test_logit = self.get_batch_train(self.length_test,0,self.test_data)
        self.logit_out = self.sess.run(self.logit_sig,feed_dict={self.input_x: self.test_data,
                                            self.init_hiddenstate:init_hidden_state})
        self.correct = 0
        for i in range(self.length_test):
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] > self.threshold:
                self.correct += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] < self.threshold:
                self.correct += 1

        self.acc = np.float(self.correct)/self.length_test


