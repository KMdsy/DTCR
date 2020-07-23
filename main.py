'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''
import os
import tensorflow as tf
import numpy as np
from utils import read_dataset, construct_classification_dataset, show_train_test_curve
from config import config_dtcr
from framework import DTCR

INDEX = '4'


os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



log_folder = 'train_log'
if os.path.exists(log_folder) == False: os.makedirs(log_folder)
img_folder = 'train_img'
if os.path.exists(img_folder) == False: os.makedirs(img_folder)

config_dtcr['indicator'] = 'RI'
NORMALIZED = True 
if __name__ == '__main__':
    
    dataset_name = 'ChlorineConcentration' # any sub-dataset in UCRArchive_2018
    
    '''dataset setting'''
    config_dtcr['train_file'] = './../UCRArchive_2018/{0}/{0}_TRAIN.tsv'.format(dataset_name) # re-config the path
    config_dtcr['test_file'] = './../UCRArchive_2018/{0}/{0}_TEST.tsv'.format(dataset_name) # re-config the path
    
    config_dtcr['img_path'] = os.path.join(img_folder, dataset_name)
    if os.path.exists(config_dtcr['img_path']) == False: os.makedirs(config_dtcr['img_path'])
    
    # load data  
    real_train_data, real_train_label, label_dict = read_dataset(config_dtcr, 'train', if_n=NORMALIZED)
    real_test_data, real_test_label, _ = read_dataset(config_dtcr, 'test', label_dict, if_n=NORMALIZED)
    
    # construct classification dataset
    cls_train_data, cls_train_label = construct_classification_dataset(real_train_data)
    
    '''dataset setting'''
    config_dtcr['input_length'] = real_train_data.shape[1]
    config_dtcr['training_samples_num'] = real_train_data.shape[0]
    config_dtcr['cluster_num'] = len(np.unique(real_train_label))

    ''' Network config searching. '''
    for encoder_config in [[100,50,50], [50,30,30]]:#[100,50,50], [50,30,30]
        config_dtcr['encoder_hidden_units'] = encoder_config

        for lambda_1 in [1, 1e-1, 1e-2, 1e-3]:#1, 1e-1, 1e-2, 1e-3
                
            config_dtcr['lambda'] = lambda_1
            
            # init the model
            print('Init the DTCR model')
            DTCR_model = DTCR(config_dtcr)
            # train the model
            print('Start training...')
            best, best_epoch, train_list, test_list = DTCR_model.train(cls_train_data, cls_train_label, real_train_data, real_train_label, real_test_data, real_test_label)
            # plot train and test curve
            show_train_test_curve(config_dtcr, train_list, test_list, INDEX)
            
            #log
            log_file = os.path.join(log_folder, '{}_log.txt'.format(dataset_name))
            if os.path.exists(log_file) == False:
                f = open(log_file, 'w')
                f.close()
            f = open(log_file, 'a')
            print('dataset: {}\trun_index: {}'.format(dataset_name, INDEX), file=f)
            print('network config:\nencoder_hidden_units = {}, lambda = {}, indicator = {}, normalized = {}'.
                format(config_dtcr['encoder_hidden_units'], config_dtcr['lambda'], config_dtcr['indicator'], NORMALIZED), file=f)
            print('best\t{} = {}, epoch = {}\n\n'.format(config_dtcr['indicator'], best, best_epoch), file=f)
            f.close()

    
    