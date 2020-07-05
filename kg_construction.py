import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import pandas as pd
#from kg_model_modify import hetero_model_modify
#from CNN_model import CNN_model
#from LSTM_model import LSTM_model
from Data_process import kg_process_data
#from Shallow_nn_ehr import NN_model
#from evaluation import cal_auc
from LSTM import LSTM_model
from Dynamic_HGM import dynamic_hgm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

class Kg_construct_ehr():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/home/tingyi/MIMIC'
        self.diagnosis = file_path + '/DIAGNOSES_ICD.csv'
        self.diagnosis_d = file_path + '/D_ICD_DIAGNOSES.csv'
        self.prescription = file_path + '/PRESCRIPTIONS.csv'
        self.charteve = file_path + '/CHARTEVENTS.csv'
        self.d_item = file_path + '/D_ITEMS.csv'
        self.noteevents = file_path + '/NOTEEVENTS.csv'
        self.proc_icd = file_path + '/PROCEDURES_ICD.csv'
        self.labeve = file_path + '/LABEVENTS.csv'
        self.patient = file_path + '/PATIENTS.csv'
        self.ICU_stay = file_path + '/ICUSTAYS.csv'
        self.input_eve = file_path + '/INPUTEVENTS_MV.csv'
        self.output_eve = file_path + '/OUTPUTEVENTS.csv'
        self.read_icustay()
        self.read_labeve()
        self.read_patient()
        self.read_diagnosis()
        self.read_charteve()
        self.read_diagnosis_d()
        self.read_prescription()
        self.read_ditem()
        #self.read_proc_icd()

    def read_inputeve(self):
        self.inputeve = pd.read_csv(self.input_eve)
        self.inputeve_ar = np.array(self.inputeve)

    def read_icustay(self):
        self.icu = pd.read_csv(self.ICU_stay)
        self.icu_ar = np.array(self.icu)

    def read_patient(self):
        self.pat = pd.read_csv(self.patient)
        self.pat_ar = np.array(self.pat)

    def read_diagnosis(self):
        self.diag = pd.read_csv(self.diagnosis)
        self.diag_ar = np.array(self.diag)

    def read_diagnosis_d(self):
        self.diag_d = pd.read_csv(self.diagnosis_d)
        self.diag_d_ar = np.array(self.diag_d)

    def read_prescription(self):
        self.pres = pd.read_csv(self.prescription)
        self.pres_ar = np.array(self.pres)

    def read_charteve(self):
        self.char = open(self.charteve)
        #self.char = pd.read_csv(self.charteve,chunksize=4000000)
        #self.char_ar = np.array(self.char.get_chunk())
        #self.num_char = self.char_ar.shape[0]

    def read_labeve(self):
        #self.lab = open(self.labeve)
        self.lab = pd.read_csv(self.labeve,chunksize=6000000)
        self.lab_ar = np.array(self.lab.get_chunk())
        self.num_lab = self.lab_ar.shape[0]

    def read_ditem(self):
        self.d_item = pd.read_csv(self.d_item)
        self.d_item_ar = np.array(self.d_item)

    def read_noteevent(self):
        self.note = pd.read_csv(self.noteevents,chunksize=1000)

    def read_proc_icd(self):
        self.proc_icd = pd.read_csv(self.proc_icd)


    def create_kg_dic(self):
        self.dic_patient = {}
        self.dic_diag = {}
        self.dic_item = {}
        self.dic_patient_addmission = {}
        index_item = 0
        index_diag = 0
        num_read = 0
        #self.icu_total = self.icu_ar[np.where(kg.icu_ar[:,-1]>1)[0]]
        self.total_data = np.intersect1d(list(self.icu_ar[:,2]),list(self.lab_ar[:,2]))
        index_count = 0
        for i in self.total_data:
            print(index_count)
            index_count += 1
            hadm_id = i
            self.index_total = i
            single_patient_hadm_lab = np.where(self.lab_ar[:,2] == i)[0]
            self.single_patient_hadm_icu = np.where(self.icu_ar[:,2] == i)
            subject_id = self.icu_ar[self.single_patient_hadm_icu][0][1]
            sub_id_pat = np.where(self.pat_ar[:,1]== subject_id)[0]
            flag = self.pat_ar[sub_id_pat][0][-1]
            in_hospital_time = self.icu_ar[self.single_patient_hadm_icu][0][-3]
            out_date_time = in_hospital_time.split(' ')
            out_date = [np.int(i) for i in out_date_time[0].split('-')]
            out_date_value = out_date[0]*365 + out_date[1]*30 + out_date[2]
            out_time = [np.int(i) for i in out_date_time[1].split(':')]
            out_time_value = out_time[0]*60 + out_time[1]
            if hadm_id not in self.dic_patient.keys():
                self.dic_patient[hadm_id] = {}
                self.dic_patient[hadm_id]['itemid'] = {}
                self.dic_patient[hadm_id]['flag'] = flag
                self.dic_patient[hadm_id]['prior_time'] = {}
            for k in single_patient_hadm_lab:
                value = self.lab_ar[k][6]
                itemid = self.lab_ar[k][3]
                if math.isnan(value):
                    continue
                self.dic_patient[hadm_id]['nodetype'] = 'patient'
                self.dic_patient[hadm_id]['next_admission'] = None
                self.dic_patient[hadm_id]['itemid'].setdefault(itemid, []).append(value)
                if itemid not in self.dic_item.keys():
                    self.dic_item[itemid] = {}
                    self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)
                    self.dic_item[itemid]['item_index'] = index_item
                    self.dic_item[itemid].setdefault('value', []).append(value)
                    index_item += 1
                else:
                    self.dic_item[itemid].setdefault('value', []).append(value)
                    if hadm_id not in self.dic_item[itemid]['neighbor_patient']:
                        self.dic_item[itemid].setdefault('neighbor_patient', []).append(hadm_id)

            for k in single_patient_hadm_lab:
                value = self.lab_ar[k][6]
                itemid = self.lab_ar[k][3]
                if math.isnan(value):
                    continue
                date_time = self.lab_ar[k][4].split(' ')
                date = [np.int(i) for i in date_time[0].split('-')]
                date_value = date[0] * 365 + date[1] * 30 + date[2]
                time = [np.int(i) for i in date_time[1].split(':')]
                time_value = time[0] * 60 + time[1]
                #if (date_value-out_date_value)>2:
                    #continue
                self.prior_time = np.int(np.floor(np.float((time_value - out_time_value)/60)))
                if self.prior_time < 0:
                    self.prior_time = 0
                if self.prior_time not in self.dic_patient[hadm_id]['prior_time']:
                    self.dic_patient[hadm_id]['prior_time'][self.prior_time] = {}
                    self.dic_patient[hadm_id]['prior_time'][self.prior_time].setdefault(itemid,[]).append(value)
                else:
                    self.dic_patient[hadm_id]['prior_time'][self.prior_time].setdefault(itemid,[]).append(value)

        for i in range(self.diag_ar.shape[0]):
            hadm_id = self.diag_ar[i][2]
            diag_icd = self.diag_ar[i][4]
            if hadm_id in self.dic_patient:
                if diag_icd not in self.dic_diag:
                    self.dic_diag[diag_icd] = {}
                    self.dic_diag[diag_icd].setdefault('neighbor_patient', []).append(hadm_id)
                    self.dic_diag[diag_icd]['nodetype'] = 'diagnosis'
                    self.dic_diag[diag_icd]['diag_index'] = index_diag
                    self.dic_patient[hadm_id].setdefault('neighbor_diag', []).append(diag_icd)
                    index_diag += 1
                else:
                    self.dic_patient[hadm_id].setdefault('neighbor_diag',[]).append(diag_icd)
                    self.dic_diag[diag_icd].setdefault('neighbor_patient',[]).append(hadm_id)

        for i in self.dic_item.keys():
            self.dic_item[i]['mean_value'] = np.mean(self.dic_item[i]['value'])
            self.dic_item[i]['std'] = np.std(self.dic_item[i]['value'])

        index_relation = 0
        for i in self.dic_item.keys():
            if not self.dic_item[i]['std'] == 0:
                self.dic_item[i]['index_relation'] = {}
                self.dic_item[i]['index_relation']['low'] = index_relation
                index_relation += 1
                self.dic_item[i]['index_relation']['middle'] = index_relation
                index_relation += 1
                self.dic_item[i]['index_relation']['high'] = index_relation
            else:
                self.dic_item[i]['index_relation'] = {}
                self.dic_item[i]['index_relation']['low'] = index_relation
                index_relation += 1





    def create_kg(self):
        self.g = nx.DiGraph()
        for i in range(self.num_char):
            patient_id = self.char_ar[i][1]/home/tingyi/ecgtoolkit-cs-git/ECGToolkit/libs/ECGConversion/MUSEXML/MUSEXML/MUSEXMLFormat.cs
            itemid = self.char_ar[i][4]
            value = self.char_ar[i][8]
            itemid_list = np.where(self.d_item_ar == itemid)
            diag_list = np.where(self.diag_ar[:,1] == patient_id)
            diag_icd9_list = self.diag_ar[:,4][diag_list]
            diag_d_list = [np.where(self.diag_d_ar[:,1] == diag_icd9_list[x])[0] for x in range(diag_icd9_list.shape[0])]
            """
            Add patient node
            """
            self.g.add_node(patient_id, item_id=itemid)
            self.g.add_node(patient_id, test_value=value)
            self.g.add_node(patient_id, node_type='patient')
            self.g.add_node(patient_id, itemid_list=itemid_list)
            self.g.add_node(itemid, node_type='ICD9')
            """
            Add diagnosis ICD9 node
            """
            self.g.add_edge(patient_id, itemid, type='')

if __name__ == "__main__":
    kg = Kg_construct_ehr()
    #kg.create_kg_dic()
    f = open("/home/tingyi/DHGM_data/dic_patient.json")
    kg.dic_patient = json.load(f)
    f.close()
    f = open("/home/tingyi/DHGM_data/dic_item.json")
    kg.dic_item = json.load(f)
    f.close()
    f = open("/home/tingyi/DHGM_data/dic_diag.json")
    kg.dic_diag = json.load(f)
    f.close()
    index_relation = 0
    kg.dic_death = {}
    #kg.dic_death[0] = {}
    #kg.dic_death[1] = {}
    for i in kg.dic_patient.keys():
        if kg.dic_patient[i]['flag'] == 0:
            kg.dic_death.setdefault(0, []).append(i)
        if kg.dic_patient[i]['flag'] == 1:
            kg.dic_death.setdefault(1, []).append(i)


    for i in kg.dic_item.keys():
         if not kg.dic_item[i]['std'] == 0:
             kg.dic_item[i]['index_relation'] = {}
             kg.dic_item[i]['index_relation']['low'] = index_relation
             index_relation += 1
             kg.dic_item[i]['index_relation']['middle'] = index_relation
             index_relation += 1
             kg.dic_item[i]['index_relation']['high'] = index_relation
             index_relation += 1
         else:
             kg.dic_item[i]['index_relation'] = {}
             kg.dic_item[i]['index_relation']['low'] = index_relation
             index_relation += 1

    process_data = kg_process_data(kg)
    process_data.seperate_train_test_cnn()
    LSTM_ = LSTM_model(kg,process_data)
    dhgm = dynamic_hgm(kg,process_data,index_relation)
