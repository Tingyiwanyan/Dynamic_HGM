import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import copy
from itertools import groupby

class dynamic_hgm():
    """
    Create dynamic HGM model
    """
    def __init__(self,kg,data_process,item_length):
        self.kg = kg
        self.data_process = data_process
        #self.hetro_model = hetro_model
        self.train_data = self.data_process.train_patient
        self.test_data = self.data_process.test_patient
        self.length_train = len(self.train_data)
        self.length_test = len(self.test_data)
        self.batch_size = 16
        self.time_sequence = 3
        self.latent_dim = 100
        self.latent_dim_cell_state = 100
        self.epoch = 6
        self.item_size = item_length #len(list(kg.dic_item.keys()))
        self.diagnosis_size = len(list(kg.dic_diag))
        self.patient_size = len(list(kg.dic_patient))
        self.input_seq = []
        self.threshold = 0.5
        self.positive_lab_size = 2
        self.negative_lab_size = 2
        self.positive_sample_size = self.positive_lab_size + 1
        self.negative_sample_size = self.negative_lab_size + 1
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

        """
        Define relation model
        """
        self.shape_relation = (self.latent_dim,)
        self.init_mortality = tf.keras.initializers.he_normal(seed=None)
        self.init_lab = tf.keras.initializers.he_normal(seed=None)
        """
        Define parameters
        """
        self.mortality = tf.placeholder(tf.float32,[None,2,2])
        self.init_weight_mortality = tf.keras.initializers.he_normal(seed=None)
        self.weight_mortality = \
            tf.Variable(self.init_weight_mortality(shape=(2,self.latent_dim)))
        self.bias_mortality = tf.Variable(self.init_weight_mortality(shape=(self.latent_dim,)))

        self.lab_test = \
            tf.placeholder(tf.float32,[None,self.positive_lab_size+self.negative_lab_size,self.item_size])
        self.weight_lab = \
            tf.Variable(self.init_weight_mortality(shape=(self.item_size,self.latent_dim)))
        self.bias_lab = tf.Variable(self.init_weight_mortality(shape=(self.latent_dim,)))
        """
        relation type 
        """
        self.relation_mortality = tf.Variable(self.init_mortality(shape=self.shape_relation))
        self.relation_lab = tf.Variable(self.init_lab(shape=self.shape_relation))

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
            if not i == 0:
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
        idx_origin = tf.constant([0])
        self.patient_lstm = tf.gather(self.hidden_last,idx_origin,axis=1)
        self.output_layer = tf.layers.dense(inputs=self.patient_lstm,
                                           units=2,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        #self.logit_sig = tf.math.sigmoid(self.output_layer)
        self.logit_sig = tf.nn.softmax(self.output_layer)
        bce = tf.keras.losses.BinaryCrossentropy()
        self.cross_entropy = bce(self.logit_sig,self.input_y_logit)
        #self.L2_norm = tf.math.square(tf.math.subtract(self.input_y_logit,self.logit_sig))
        #self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.L2_norm,axis=1),axis=0)

    def build_dhgm_model(self):
        """
        Build dynamic HGM model
        """
        self.Dense_patient = tf.expand_dims(self.hidden_last,1)
        #self.Dense_patient = self.hidden_last

        self.Dense_mortality_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.mortality,self.weight_mortality),self.bias_mortality))

        self.Dense_mortality = tf.math.subtract(self.Dense_mortality_,self.relation_mortality)

        self.Dense_lab_ = \
            tf.nn.relu(tf.math.add(tf.matmul(self.lab_test,self.weight_lab),self.bias_lab))

        self.Dense_lab = tf.math.add(self.Dense_lab_,self.relation_lab)

    def get_latent_rep_hetero(self):
        """
        Prepare data for SGNS loss function
        """
        idx_origin = tf.constant([0])
        self.x_origin = tf.gather(self.Dense_patient,idx_origin,axis=1)
        #self.x_origin = self.hidden_last

        idx_skip_mortality = tf.constant([0])
        self.x_skip = tf.gather(self.Dense_mortality,idx_skip_mortality,axis=1)
        idx_neg_mortality = tf.constant([1])
        self.x_negative = tf.gather(self.Dense_mortality,idx_neg_mortality,axis=1)


        item_idx_skip = tf.constant([i+1 for i in range(self.positive_lab_size)])
        self.x_skip_item = tf.gather(self.Dense_patient,item_idx_skip,axis=1)
        item_idx_negative = tf.constant([i+self.positive_lab_size+1 for i in range(self.negative_lab_size)])
        self.x_negative_item = tf.gather(self.Dense_patient,item_idx_negative,axis=1)

        self.x_skip = tf.concat([self.x_skip,self.x_skip_item],axis=1)
        self.x_negative = tf.concat([self.x_negative,self.x_negative_item],axis=1)


    def get_positive_patient(self,center_node_index):
        self.patient_pos_sample = np.zeros((self.positive_lab_size,self.time_sequence, self.item_size))
        if self.kg.dic_patient[center_node_index]['flag'] == 0:
            neighbor_patient = self.kg.dic_death[0]
        else:
            neighbor_patient = self.kg.dic_death[1]
        for i in range(self.positive_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            time_index = 0
            for j in time_seq_int:
                if time_index == self.time_sequence:
                    break
                #self.time_index = np.int(j)
                one_data = self.assign_value_patient(patient_id,j)
                self.patient_pos_sample[i,time_index,:] = one_data

    def get_negative_patient(self,center_node_index):
        self.patient_neg_sample = np.zeros((self.negative_lab_size,self.time_sequence, self.item_size))
        if self.kg.dic_patient[center_node_index]['flag'] == 0:
            neighbor_patient = self.kg.dic_death[1]
        else:
            neighbor_patient = self.kg.dic_death[0]
        for i in range(self.negative_lab_size):
            index_neighbor = np.int(np.floor(np.random.uniform(0, len(neighbor_patient), 1)))
            patient_id = neighbor_patient[index_neighbor]
            time_seq = self.kg.dic_patient[patient_id]['prior_time'].keys()
            time_seq_int = [np.int(k) for k in time_seq]
            time_seq_int.sort()
            time_index = 0
            for j in time_seq_int:
                if time_index == self.time_sequence:
                    break
                #self.time_index = np.int(j)
                one_data = self.assign_value_patient(patient_id,j)
                self.patient_neg_sample[i,time_index,:] = one_data

    def get_positive_samples(self,center_node_index,time):
        self.pos_nodes_item = []
        self.check_center_node = center_node_index
        self.item_pos_sample = np.zeros((self.positive_lab_size,self.item_size))
        index = 0
        """
        get pos set for item
        """
        #item_neighbor_nodes = self.kg.dic_patient[center_node_index]['itemid'].keys()
        item_neighbor_nodes = self.kg.dic_patient[center_node_index]['prior_time'][str(time)].keys()
        for j in range(self.positive_lab_size):
            index_item = np.int(np.floor(np.random.uniform(0,len(item_neighbor_nodes),1)))
            item_neighbor = item_neighbor_nodes[index_item]
            self.pos_nodes_item.append(item_neighbor_nodes[index_item])
            one_sample_pos_item = self.assign_value_item(center_node_index,item_neighbor,time)
            self.item_pos_sample[index,:] = one_sample_pos_item
            index += 1

    """
    def get_positive_sample_rep(self):
        self.item_pos_sample = np.zeros((self.positive_lab_size,self.item_size))
        index = 0
        for i in self.pos_nodes_item:
            one_sample_pos_item = self.assign_value_item(i)
            self.item_pos_sample[index,:] = one_sample_pos_item
            index += 1
    """

    def get_negative_samples(self,center_node_index,time):
        self.neg_nodes_item = []
        self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        """
        get neg set for item
        """
        item_neighbor_nodes_whole = self.kg.dic_patient[center_node_index]['itemid'].keys()
        item_neighbor_nodes = self.kg.dic_patient[center_node_index]['prior_time'][str(time)].keys()
        whole_item_nodes = self.kg.dic_item.keys()
        neg_set_item = [i for i in whole_item_nodes if i not in item_neighbor_nodes_whole]
        index_item_count = 0
        for j in range(self.negative_lab_size):
            index_item = np.int(np.floor(np.random.uniform(0, len(item_neighbor_nodes), 1)))
            item_neighbor = item_neighbor_nodes[index_item]
            if len(self.kg.dic_item[item_neighbor]['index_relation'].keys()) > 1:
                one_sample_neg_item = self.assign_value_item_neg(center_node_index, item_neighbor, time)
            else:
                #print("im here")
                index_sample = np.int(np.floor(np.random.uniform(0,len(neg_set_item),1)))
                neg_node = neg_set_item[index_sample]
                one_sample_neg_item = self.assign_value_item_neg_whole(neg_node)
                #self.neg_nodes_item.append(neg_set_item[index_sample])
            self.item_neg_sample[index_item_count,:] = one_sample_neg_item
            index_item_count += 1
    """
    def get_negative_sample_rep(self):
        self.item_neg_sample = np.zeros((self.negative_lab_size,self.item_size))
        index = 0
        for i in self.neg_nodes_item:
            one_sample_neg_item = self.assign_value_item(i)
            self.item_neg_sample[index,:] = one_sample_neg_item
            index += 1
    """


    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.positive_sample_size, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))


    def config_model(self):
        self.lstm_cell()
        self.softmax_loss()
        self.build_dhgm_model()
        self.get_latent_rep_hetero()
        self.SGNN_loss()
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
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


    def assign_value_item(self,center_node_index,itemid,time):
        one_sample = np.zeros(self.item_size)
        #index = self.kg.dic_item[itemid]['item_index']
        ave_value = np.mean(self.kg.dic_patient[center_node_index]['prior_time'][str(time)][itemid])
        mean = self.kg.dic_item[itemid]['mean_value']
        std = self.kg.dic_item[itemid]['std']
        if std == 0:
            index = self.kg.dic_item[itemid]['index_relation']['low']
        else:
            if np.abs(np.float(ave_value)-mean) < std:
                #print("low")
                index = self.kg.dic_item[itemid]['index_relation']['low']
            elif np.abs(np.float(ave_value)-mean) < 2*std:
                #print("middle")
                index = self.kg.dic_item[itemid]['index_relation']['middle']
            else:
                #print("high")
                index = self.kg.dic_item[itemid]['index_relation']['high']
        one_sample[index] = 1

        return one_sample

    def assign_value_item_neg(self,center_node_index,itemid,time):
        one_sample = np.zeros(self.item_size)
        #index = self.kg.dic_item[itemid]['item_index']
        ave_value = np.mean(self.kg.dic_patient[center_node_index]['prior_time'][str(time)][itemid])
        mean = self.kg.dic_item[itemid]['mean_value']
        std = self.kg.dic_item[itemid]['std']
        if np.abs(np.float(ave_value)-mean) < std:
            index = self.kg.dic_item[itemid]['index_relation']['high']
        elif np.abs(np.float(ave_value)-mean) < 2*std:
            random_ind = np.int(np.floor(np.random.uniform(0, 2, 1)))
            if random_ind == 0:
                index = self.kg.dic_item[itemid]['index_relation']['low']
            else:
                index = self.kg.dic_item[itemid]['index_relation']['high']
        else:
            index = self.kg.dic_item[itemid]['index_relation']['low']
        one_sample[index] = 1

        return one_sample

    def assign_value_item_neg_whole(self,itemid):
        one_sample = np.zeros(self.item_size)
        std = self.kg.dic_item[itemid]['std']
        if std == 0:
            index = self.kg.dic_item[itemid]['index_relation']['low']
        else:
            random_ind = np.int(np.floor(np.random.uniform(0, 3, 1)))
            if random_ind == 0:
                index = self.kg.dic_item[itemid]['index_relation']['low']
            elif random_ind == 1:
                index = self.kg.dic_item[itemid]['index_relation']['middle']
            else:
                index = self.kg.dic_item[itemid]['index_relation']['high']
        one_sample[index] = 1

        return one_sample

    def get_batch_train(self,data_length,start_index,data):
        """
        get training batch data
        """
        train_one_batch = np.zeros((data_length, self.time_sequence,self.item_size))
        train_one_batch_item = np.zeros((data_length,self.positive_lab_size+self.negative_lab_size,self.item_size))
        train_one_batch_mortality = np.zeros((data_length,2,2))
        one_batch_logit = np.zeros((data_length,2))
        self.real_logit = np.zeros(data_length)
        #self.item_neg_sample = np.zeros((self.negative_lab_size, self.item_size))
        #self.item_pos_sample = np.zeros((self.positive_lab_size, self.item_size))
        index_batch = 0
        index_increase = 0
        #while index_batch < data_length:
        for i in range(data_length):
            self.patient_id = data[start_index + i]
            #if self.kg.dic_patient[self.patient_id]['item_id'].keys() == {}:
             #   index_increase += 1
              #  continue
            #index_batch += 1
            self.time_seq = self.kg.dic_patient[self.patient_id]['prior_time'].keys()
            self.time_seq_int = [np.int(k) for k in self.time_seq]
            self.time_seq_int.sort()
            time_index = 0
            flag = self.kg.dic_patient[self.patient_id]['flag']
            """
            if flag == 0:
                one_batch_logit[i,0,0] = 1
                one_batch_logit[i,1,1] = 1
            else:
                one_batch_logit[i,0,1] = 1
                one_batch_logit[i,1,0] = 1
                self.real_logit[i] = 1
            """
            if flag == 0:
                train_one_batch_mortality[i,0,:] = [1,0]
                train_one_batch_mortality[i,1,:] = [0,1]
                one_batch_logit[i, 0] = 1
            else:
                train_one_batch_mortality[i,0,:] = [0,1]
                train_one_batch_mortality[i,1,:] = [1,0]
                one_batch_logit[i, 1] = 1
            """
            self.get_positive_samples(self.patient_id)
            self.get_positive_sample_rep()
            self.get_negative_samples(self.patient_id)
            self.get_negative_sample_rep()
            """
            if len(self.time_seq_int) < self.time_sequence:
                time_item_assign = self.time_seq_int[-1]
            else:
                time_item_assign = self.time_seq_int[self.time_sequence-1]
            self.get_positive_samples(self.patient_id,time_item_assign)
            self.get_negative_samples(self.patient_id,time_item_assign)
            single_one_item = np.concatenate((self.item_pos_sample,self.item_neg_sample))
            train_one_batch_item[i,:,:] = single_one_item

            #self.get_positive_patient(self.patient_id)
            #self.get_negative_patient(self.patient_id)
            #train_one_batch[i,1:1+self.positive_lab_size,:,:] = self.patient_pos_sample
            #train_one_batch[i,1+self.positive_lab_size:,:,:] = self.patient_neg_sample
            for j in self.time_seq_int:
                if time_index == self.time_sequence:
                    break
                #self.time_index = np.int(j)
                self.one_data = self.assign_value_patient(self.patient_id,j)
                train_one_batch[i,time_index,:] = self.one_data
                time_index += 1

        return train_one_batch, one_batch_logit, train_one_batch_mortality,train_one_batch_item

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
                self.train_one_batch, self.one_batch_logit,self.one_batch_mortality,self.one_batch_item = self.get_batch_train(self.batch_size,i*self.batch_size,self.train_data)

                self.err_ = self.sess.run([self.negative_sum, self.train_step_neg],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.lab_test: self.one_batch_item,
                                                self.mortality: self.one_batch_mortality,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_[0])

                """
                self.err_lstm = self.sess.run([self.cross_entropy, self.train_step_cross_entropy,self.init_hiddenstate,self.output_layer,self.logit_sig],
                                     feed_dict={self.input_x: self.train_one_batch,
                                                self.input_y_logit: self.one_batch_logit,
                                                self.init_hiddenstate:init_hidden_state})
                print(self.err_lstm[0])
                """


    def test(self):
        init_hidden_state = np.zeros((self.length_test, self.latent_dim))
        self.test_data_batch, self.test_logit, self.test_mortality,self.test_item = self.get_batch_train(self.length_test, 0, self.test_data)
        self.test_patient = self.sess.run(self.hidden_last, feed_dict={self.input_x: self.test_data_batch,
                                                                  self.init_hiddenstate: init_hidden_state})
        single_mortality = np.zeros((1,2,2))
        single_mortality[0][0][0] = 1
        single_mortality[0][1][1] = 1
        self.mortality_test = self.sess.run(self.Dense_mortality,feed_dict={self.mortality:single_mortality})[0]
        self.score = np.zeros(self.length_test)
        for i in range(self.length_test):
            embed_single_patient = self.test_patient[i]/np.linalg.norm(self.test_patient[i])
            embed_mortality = self.mortality_test[1]/np.linalg.norm(self.mortality_test[1])
            self.score[i] = np.matmul(embed_single_patient,embed_mortality.T)

        self.correct = 0
        for i in range(self.length_test):
            if self.score[i]<0 and self.test_logit[i,0] == 1:
                self.correct += 1
            if self.score[i]>0 and self.test_logit[i,1] == 1:
                self.correct += 1



    def test_lstm(self):
        """
        test the system, return the accuracy of the model
        """
        init_hidden_state = np.zeros((self.length_test, self.latent_dim))
        self.test_data, self.test_logit,self.train_one_batch_item = self.get_batch_train(self.length_test,0,self.test_data)
        self.logit_out = self.sess.run(self.logit_sig,feed_dict={self.input_x: self.test_data,
                                            self.init_hiddenstate:init_hidden_state})
        self.correct = 0
        for i in range(self.length_test):
            if self.test_logit[i,1] == 1 and self.logit_out[i,1] > self.threshold:
                self.correct += 1
            if self.test_logit[i,1] == 0 and self.logit_out[i,1] < self.threshold:
                self.correct += 1

        self.acc = np.float(self.correct)/self.length_test
