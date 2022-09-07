#!/usr/bin/env python

import os, warnings, sys 
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split


import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils

#import algorithm.scoring as scoring
from algorithm.model.recommender import Recommender, get_data_based_model_params
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
    
    # normally we would do train, valid split using the given data, but we are not doing that here
    # because we need the total num_users and num_items from the full train data
    # also, the target will be scaled, but with ratings, we can assume the scale of data will 
    # remain the same in the future
    # so we can use the same scaler for train/valid/test datasets
    train_data = data
    # print('train_data shape:',  train_data.shape)  
    
    # preprocess data
    print("Pre-processing data...")
    train_data, _, preprocess_pipe = preprocess_data(train_data, None, data_schema)  
    train_X, train_y = train_data['X'], train_data['y']  
    # print('train_X/y shape:',  train_X.shape, train_y.shape)     
              
    # Create and train model     
    print('Fitting model ...')  
    model, history = train_model(train_X, train_y, hyper_params, verbose=1)    
    
    return preprocess_pipe, model, history


def train_model(train_X, train_y, hyper_params, verbose=0):   
    # get model hyper-parameters  that are data-dependent (in this case N = num_users, and M = num_items)    
    data_based_params = get_data_based_model_params(train_X) 
    
    model_params = { **data_based_params, **hyper_params }
    
    # Create and train model   
    model = Recommender(  **model_params )  
    # model.summary()  
    
    # fit model
    history = model.fit( 
            X=train_X, 
            y=train_y, 
            validation_split=model_cfg["valid_split"],
            epochs=100,
            verbose=verbose,
        )  
    
    return model, history


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(data_schema)   
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
      
    if valid_data is not None:
        valid_data = preprocess_pipe.transform(valid_data)
        # print("Processed valid X/y data shape", valid_data['X'].shape, valid_data['y'].shape)
    return train_data, valid_data, preprocess_pipe 


