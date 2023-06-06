#!/usr/bin/env python
# coding: utf-8
## @@author sakib570
## Parts of the code are reused from https://github.com/navodas/MIA with modifications

# In[ ]:

import tensorflow.keras as keras
import numpy as np
from sklearn.utils import resample
import pandas as pd
import pickle
import os
import csv
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
MODEL_PATH = './model/'
DATA_PATH = './data/'


# In[ ]:


def read_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x, train_y, test_x, test_y


# In[ ]:


def transform_adult_data(dataset, is_synthetic):
    
    if(is_synthetic == False):
        for col in [1,3,5,6,7,8,9,13,14]:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype('str'))

        # normalize the values
        x_range = [i for i in range(14)]
        dataset[x_range] = dataset[x_range]/dataset[x_range].max()

        x = dataset[x_range].values
        y = dataset[14].values
    else:
        for col in [1,2,3,4,5,6,7,11,12]:
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col].astype('str'))

        # normalize the values
        x_range = [i for i in range(12)]
        dataset[x_range] = dataset[x_range]/dataset[x_range].max()

        x = dataset[x_range].values
        y = dataset[12].values
        
    
    dim = x.shape[1]
    
    x=np.array(x)
    y=np.array(y)
    
    
    y=to_categorical(y)
    
    return x, y, dim

def transform_location_data(dataset): 
    df_tot = dataset
    df_tot.dropna(inplace=True)

    trainX = df_tot.iloc[:,1:]
    trainY = df_tot.iloc[:,0]
    

    dim=trainX.shape[1]


    #num of classes
    num_classes=30

    trainX=np.array(trainX)
    trainY=np.array(trainY)
    
    
    trainY = to_categorical(trainY)

    return trainX, trainY, dim


def transform_purchase_data(dataset): 
    df_tot = dataset
    df_tot.dropna(inplace=True)

    trainX = df_tot.iloc[:,0:dataset.shape[1]-1]
    trainY = df_tot.iloc[:,-1]

    dim=trainX.shape[1]


    #num of classes
    num_classes=100

    trainX=np.array(trainX)
    trainY=np.array(trainY)
    
    trainY = to_categorical(trainY)


    return trainX, trainY, dim

def transform_avila_data(dataset, is_synthetic):
    

    le = LabelEncoder()
    dataset[10] = le.fit_transform(dataset[10].astype('str'))

    # normalize the values
    x_range = [i for i in range(10)]
    #dataset[x_range] = dataset[x_range]/dataset[x_range].max()


    x = dataset[x_range].values
    y = dataset[10].values
        
    
    dim = x.shape[1]
    
    x=np.array(x)
    y=np.array(y)

    y=to_categorical(y)
    
    return x, y, dim

def transform_polish_data(dataset):
    
    for col in [0,2,3,4,5,6,7,8,9,10,13]:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col].astype('str'))
    for col in [1,11,12]:
        dataset[col] = dataset[col].astype('int')

    # normalize the values
    x_range = [i for i in range(13)]
    dataset[x_range] = dataset[x_range]/dataset[x_range].max()

    x = dataset[x_range].values
    y = dataset[13].values
    
    dim = x.shape[1]
    
    x=np.array(x)
    y=np.array(y)
    
    
    y=to_categorical(y)
    
    return x, y, dim
# In[ ]:


def load_target_data(dataset, dataset_name, train_size, test_start, test_ratio, is_synthetic):
    
    if dataset_name == 'adult':
        x, y, dim = transform_adult_data(dataset, is_synthetic)
    elif dataset_name == 'purchase':
        x, y, dim = transform_purchase_data(dataset)
    elif dataset_name == 'location':
        x, y, dim = transform_location_data(dataset)
    elif dataset_name == 'avila':
        x, y, dim = transform_avila_data(dataset, is_synthetic)
    elif dataset_name == 'polish':
        x, y, dim = transform_polish_data(dataset)
    
    #trainX,testX, trainY, testY = train_test_split(x, y, test_size=test_ratio, random_state=0, stratify=y)
    trainX = x[0:train_size,]
    testX = x[test_start:,]
    trainY = y[0:train_size,]
    testY = y[test_start:,]
    
    return (trainX, trainY), (testX, testY), dim


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def build_simple_mlp(n_class,pix,d):

    model = Sequential()
    model.add(Dense(256, input_dim=pix))
    model.add(Activation("relu"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(256, kernel_regularizer=l2(0.01)))
    model.add(Activation("relu"))
    #model.add(Dropout(0.01))
    
    
    #model.add(Dense(248))
    #model.add(Activation("relu"))
    #model.add(Dropout(0.01))

    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.01))
    
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    act_layer=3
    
    return model, act_layer


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def build_dnn(n_class,dim):
    model = Sequential()
    
    model.add(Dense(600, input_dim=dim))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    #model.add(Dense(1024, kernel_regularizer=l2(0.00003)))
    #model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(512, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))

    model.add(Dense(256, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(128, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    #opt = SGD(lr=0.01, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    act_layer=6
    
    return model, act_layer

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def build_location_dnn(n_class,dim):
    model = Sequential()
    
    model.add(Dense(512, input_dim=dim, kernel_regularizer=l2(0.0007)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(248, kernel_regularizer=l2(0.0007)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(128, kernel_regularizer=l2(0.0007)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))

    model.add(Dense(64, kernel_regularizer=l2(0.0007)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
       
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    act_layer=6
    
    return model, act_layer

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def build_purchase_dnn(n_class,dim):
    model = Sequential()
    
    model.add(Dense(600, input_dim=dim))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    #model.add(Dense(1024), kernel_regularizer=l2(0.001))
    #model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(512, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))

    model.add(Dense(256, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(128, kernel_regularizer=l2(0.00003)))
    model.add(Activation("tanh"))
    #model.add(Dropout(0.01))
    
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    #opt = SGD(lr=0.01, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    act_layer=6
    
    return model, act_layer


# In[ ]:


def load_shadow_data(dataset, dataset_name, n_shadow, shadow_size, test_ratio, is_synthetic):
    
    if dataset_name == 'adult':
        x, y, _ = transform_adult_data(dataset, is_synthetic)
    elif dataset_name == 'purchase':
        x, y, _ = transform_purchase_data(dataset)
    elif dataset_name == 'location':
        x, y, _ = transform_location_data(dataset)
    elif dataset_name == 'avila':
        x, y, _ = transform_avila_data(dataset, is_synthetic)
    elif dataset_name == 'polish':
        x, y, _ = transform_polish_data(dataset)
    
    shadow_indices = np.arange(len(dataset))
    
   
    for i in range(n_shadow):
        shadow_i_indices = np.random.choice(shadow_indices, shadow_size, replace=False)
        shadow_i_x, shadow_i_y = x[shadow_i_indices], y[shadow_i_indices]
        trainX,testX, trainY, testY = train_test_split(shadow_i_x, shadow_i_y, test_size=test_ratio)
        #print('shadow_i_trainX = ', len(trainX), 'shadow_i_trainY = ', len(trainY), 'shadow_i_testX = ', len(testX), 'shadow_i_testY = ', len(testY))
        
        np.savez(DATA_PATH + 'shadow_' + dataset_name + '{}_data.npz'.format(i), trainX, trainY, testX, testY)


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with modifications
def train_shadow_models(dataset_name, n_shadow, n_class, dim, epochs, channel):
    full_sm_train_pred=[]
    full_sm_train_class=[]
    
    full_sm_test_pred=[]
    full_sm_test_class=[]
    
    full_clz_train=[]
    full_clz_test=[]
    
    members=[]
    nonmembers=[]
    
    train_accuracy=[]
    test_accuracy=[]

    for j in range(n_shadow):
        
        print("Shadow Model ", j)
        
        print('Training shadow model {}'.format(j))
        data = read_data('shadow_' + dataset_name + '{}_data.npz'.format(j))
        x_shadow_train, y_shadow_train, x_shadow_test, y_shadow_test = data
        #print('x_shadow trian\n', x_shadow_train,'\n y_shadow trian\n', y_shadow_train, '\n x_shadow test\n', x_shadow_test, '\n y_shadow test\n', y_shadow_test)

        #model, act_layer = build_simple_mlp(n_class,dim, channel)
        if dataset_name == 'adult':
            model, act_layer = build_dnn(n_class,dim)
        elif dataset_name == 'purchase':
            model, act_layer = build_purchase_dnn(n_class,dim)
        elif dataset_name == 'location':
            model, act_layer = build_location_dnn(n_class,dim)
        elif dataset_name == 'avila':
            model, act_layer = build_dnn(n_class,dim)
        elif dataset_name == 'polish':
            model, act_layer = build_dnn(n_class,dim)
            
            
        # fit model
        history = model.fit(x_shadow_train, y_shadow_train, epochs=epochs, batch_size=32, validation_data=(x_shadow_test, y_shadow_test), verbose=0)
    
        # evaluate model
        _, train_acc = model.evaluate(x_shadow_train, y_shadow_train, verbose=0)
        _, test_acc = model.evaluate(x_shadow_test, y_shadow_test, verbose=0)
        print("Shadow Train acc : ", (train_acc * 100.0),"Shadow Test acc : ", (test_acc * 100.0))
        train_accuracy.append((train_acc * 100.0))
        test_accuracy.append((test_acc * 100.0))

    
        #train SM
        sm_train_pred=model.predict(x_shadow_train, batch_size=32)
        sm_train_class=np.argmax(y_shadow_train,axis=1)
    
    
        #test SM
        sm_test_pred=model.predict(x_shadow_test, batch_size=32)
        sm_test_class=np.argmax(y_shadow_test,axis=1)
        
     
        full_sm_train_pred.append(sm_train_pred)        
        full_sm_train_class.append(sm_train_class)
        members.append(np.ones(len(sm_train_pred)))
        
        full_sm_test_pred.append(sm_test_pred)        
        full_sm_test_class.append(sm_test_class) 
        nonmembers.append(np.zeros(len(sm_test_pred)))


    full_sm_train_pred = np.vstack(full_sm_train_pred)
    full_sm_train_class = [item for sublist in full_sm_train_class for item in sublist]
    members = [item for sublist in members for item in sublist]
    
    full_sm_test_pred = np.vstack(full_sm_test_pred)
    full_sm_test_class = [item for sublist in full_sm_test_class for item in sublist]
    nonmembers = [item for sublist in nonmembers for item in sublist]
    
    shadow_train_performance=(full_sm_train_pred, np.array(full_sm_train_class))
    shadow_test_performance=(full_sm_test_pred, np.array(full_sm_test_class))


    ###atack data preparation
    attack_x = (full_sm_train_pred,full_sm_test_pred)
    #attack_x = np.vstack(attack_x)
    
    attack_y = (np.array(members).astype('int32'),np.array(nonmembers).astype('int32'))
    #attack_y = np.concatenate(attack_y)
    #attack_y = attack_y.astype('int32')
    
    
    classes = (np.array(full_sm_train_class),np.array(full_sm_test_class))
    #classes = np.array([item for sublist in classes for item in sublist])


    attack_dataset = (attack_x,attack_y,classes)
    shadow_accuracy = (train_accuracy, test_accuracy)

            
    return  shadow_train_performance, shadow_test_performance, attack_dataset, x_shadow_train, y_shadow_train, x_shadow_test, y_shadow_test, model, shadow_accuracy


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def define_attack_model(n_class):
    model = Sequential()
    
    model.add(Dense(1))
    model.add(Activation("relu"))
    
    model.add(Dense(1))
    model.add(Activation("relu"))

    model.add(Dense(n_class, activation='softmax'))

    # compile model
    opt = SGD(learning_rate=0.0001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


def attack_mlp(pix,d):

    model = Sequential()
    model.add(Dense(64, input_dim=pix))
    model.add(Activation("relu"))
    #model.add(Dropout(0.1))

#     model.add(Dense(32))
#     model.add(Activation("tanh"))
#     model.add(Dropout(0.01))
    
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    act_layer=1
    
    return model, act_layer


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def prep_attack_train_data(n_attack_data):

    attack_mem = pd.DataFrame(n_attack_data[0][0])
    attack_nmem = pd.DataFrame(n_attack_data[0][1])
    
    attack_mem_status = pd.DataFrame(n_attack_data[1][0])
    attack_mem_status.columns = ["membership"]
    
    attack_nmem_status = pd.DataFrame(n_attack_data[1][1])
    attack_nmem_status.columns = ["membership"]
    
    real_class_mem = pd.DataFrame(n_attack_data[2][0])
    real_class_mem.columns = ["y"]
    
    real_class_nmem = pd.DataFrame(n_attack_data[2][1])
    real_class_nmem.columns = ["y"]

    memdf = pd.concat([attack_mem,attack_nmem],axis=0)
    memdf = memdf.reset_index(drop=True)

    memstatus =  pd.concat([attack_mem_status,attack_nmem_status],axis=0)
    memstatus = memstatus.reset_index(drop=True)

    realclass = pd.concat([real_class_mem,real_class_nmem],axis=0)
    realclass = realclass.reset_index(drop=True)

    attack_df = pd.concat([memdf,realclass,memstatus],axis=1)
    
    return attack_df


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def prep_validation_data(attack_test_data):

    attack_mem = pd.DataFrame(attack_test_data[0][0])
    attack_nmem = pd.DataFrame(attack_test_data[0][1])
    
    attack_mem_status = pd.DataFrame(attack_test_data[1][0])
    attack_mem_status.columns = ["membership"]
    
    attack_nmem_status = pd.DataFrame(attack_test_data[1][1])
    attack_nmem_status.columns = ["membership"]
    
    real_class_mem = pd.DataFrame(attack_test_data[2][0])
    real_class_mem.columns = ["y"]
    
    real_class_nmem = pd.DataFrame(attack_test_data[2][1])
    real_class_nmem.columns = ["y"]
    
    mem_df = pd.concat([attack_mem,real_class_mem],axis=1)
    nmem_df = pd.concat([attack_nmem,real_class_nmem],axis=1)

#     memdf = pd.concat([attack_mem,attack_nmem],axis=0)
#     memdf = memdf.reset_index(drop=True)

#     memstatus =  pd.concat([attack_mem_status,attack_nmem_status],axis=0)
#     memstatus = memstatus.reset_index(drop=True)

#     realclass = pd.concat([real_class_mem,real_class_nmem],axis=0)
#     realclass = realclass.reset_index(drop=True)

#     attack_df = pd.concat([memdf,realclass,memstatus],axis=1)
    
    return mem_df, nmem_df


# In[ ]:


def load_attack_test_data(dataset_name, members, nonmembers, is_synthetic):
    
    if dataset_name == 'adult':
        memberX, memberY, _ = transform_adult_data(members, is_synthetic)
        nonmemberX, nonmemberY, _ = transform_adult_data(nonmembers, is_synthetic)
    elif dataset_name == 'purchase':
        memberX, memberY, _ = transform_purchase_data(members)
        nonmemberX, nonmemberY, _ = transform_purchase_data(nonmembers)
    elif dataset_name == 'location':
        memberX, memberY, _ = transform_location_data(members)
        nonmemberX, nonmemberY, _ = transform_location_data(nonmembers)
    elif dataset_name == 'avila':
        memberX, memberY, _ = transform_avila_data(members, is_synthetic)
        nonmemberX, nonmemberY, _ = transform_avila_data(nonmembers, is_synthetic)
    elif dataset_name == 'polish':
        memberX, memberY, _ = transform_polish_data(members)
        nonmemberX, nonmemberY, _ = transform_polish_data(nonmembers)
    
    
    #memberX, memberY, _ = transform_data(members, is_synthetic)
    
    #nonmemberX, nonmemberY, _ = transform_data(nonmembers, is_synthetic)
    
    return memberX, memberY, nonmemberX, nonmemberY


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with minor modifications
def prety_print_result(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    print('TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    if tp == fp == 0:
        print('PPV: 0\nAdvantage: 0')
    else:
        print('PPV: %.4f\nAdvantage: %.4f' % (tp / (tp + fp), tp / (tp + fn) - fp / (tn + fp)))

    return tp, fp, fn, tn, (tp / (tp + fp)), (tp / (tp + fn) - fp / (tn + fp)), ((tp+tn)/(tp+tn+fp+fn)),  (tp / (tp + fn))


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with modifications
def train_attack_model(attack_data, check_membership, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50, model='nn', l2_ratio=1e-7):

    x, y,  classes = attack_data

    train_x = x[0]
    train_y = y[0]
    test_x = x[1]
    test_y = y[1]
    train_classes = classes[0]
    test_classes = classes[1]
    
    
    checkmem_prediction_vals, checkmem_membership_status, checkmem_class_status = check_membership
    
    checkmem_prediction_vals=np.vstack(checkmem_prediction_vals)
    checkmem_membership_status=np.array([item for sublist in checkmem_membership_status for item in sublist])
    checkmem_class_status=np.array([item for sublist in checkmem_class_status for item in sublist])
    
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)


    predicted_membership, target_membership = [], []
    for c in unique_classes:
        print("Class : ", c)
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)        
        
        full_cx_data=(c_train_x,c_test_x)
        full_cx_data = np.vstack(full_cx_data)

        full_cy_data=(c_train_y,c_test_y)
        full_cy_data = np.array([item for sublist in full_cy_data for item in sublist])
        
        d=1
        pix = full_cx_data.shape[1]
        classifier, _ = attack_mlp(pix,d)
        history = classifier.fit(full_cx_data, full_cy_data, epochs=EPS, batch_size=32, verbose=0)

        #get predictions on real train and test data
        c_indices = np.where(checkmem_class_status==c)
        pred_y = classifier.predict(checkmem_prediction_vals[c_indices])
        print(pred_y)
        c_pred_y = np.argmax(pred_y, axis=1)
        c_target_y = checkmem_membership_status[c_indices]
        
       
        target_membership.append(c_target_y)
        predicted_membership.append(c_pred_y)

    target_membership=np.array([item for sublist in target_membership for item in sublist])
    predicted_membership=np.array([item for sublist in predicted_membership for item in sublist])


    tp, fp, fn, tn, precision, advj, acc, recall = prety_print_result (target_membership,predicted_membership)   
    return tp, fp, fn, tn, precision, advj, acc, recall


# In[ ]:

## @@author navodas
## Reused from https://github.com/navodas/MIA with modifications
def shokri_attack(attack_df, mem_validation, nmem_validation, epochs):
    
    predicted_membership, predicted_nmembership, true_membership, TP_idx, TN_idx , mpred_all, nmpred_all = [], [], [], [], [], [], []

    class_val = np.unique(attack_df['y'])
    ncval=attack_df.shape[1]-1
    
    for c_val in class_val:

        print(c_val)
        
        filter_rec_all = attack_df[(attack_df['y'] == c_val)]
        filter_rec_idx = np.array(filter_rec_all.index)
        
        attack_feat = filter_rec_all.iloc[:, 0:ncval]
        attack_class = filter_rec_all['membership']
             
        d=1
        pix = attack_feat.shape[1]
        
        attack_model, _ = attack_mlp(pix,d)
        
       
        history = attack_model.fit(attack_feat, attack_class, epochs=epochs, batch_size=32, verbose=0)
        
        mcval=mem_validation.shape[1]-1
        
        
        check_mem_feat = mem_validation[mem_validation['y']==c_val]
        check_nmem_feat = nmem_validation[nmem_validation['y']==c_val]
        
        if (len(check_mem_feat)!=0) and (len(check_nmem_feat)!=0):
        
            check_mem_feat_idx =  np.array(check_mem_feat.index)


            check_nmem_feat_idx =  np.array(check_nmem_feat.index)

            #print(check_nmem_feat_idx)
            #print(np.argmax(mpred,axis=1)==0)


            mpred = attack_model.predict(np.array(check_mem_feat))
            predicted_membership.append(np.argmax(mpred,axis=1) )
            mpred_all.append(mpred)
            

            nmpred = attack_model.predict(np.array(check_nmem_feat))    
            predicted_nmembership.append(np.argmax(nmpred,axis=1) )
            nmpred_all.append(nmpred)
            


            TP_idx.append(check_mem_feat_idx[np.where(np.argmax(mpred,axis=1)==1)[0]])

            TN_idx.append(check_nmem_feat_idx[np.where(np.argmax(nmpred,axis=1)==0)[0]])

    pred_members = np.array([item for sublist in predicted_membership for item in sublist])
    pred_nonmembers = np.array([item for sublist in predicted_nmembership for item in sublist])
    
    TP_idx_list = np.array([item for sublist in TP_idx for item in sublist])
    TN_idx_list = np.array([item for sublist in TN_idx for item in sublist])
    
    members=np.array(list(pred_members))
    nonmembers=np.array(list(pred_nonmembers))
    
    pred_membership = np.concatenate([members,nonmembers])
    ori_membership = np.concatenate([np.ones(len(members)), np.zeros(len(nonmembers))])
    
    return pred_membership, ori_membership, TP_idx_list, TN_idx_list, mpred_all, nmpred_all


# In[ ]:


def train_target_model(target_dataset, dataset_name, per_class_sample, epoch, n_class, train_size, test_start, is_synthetic = False, channel=0, verbose=0, test_ratio=0.3):
    
    (target_trainX, target_trainY), (target_testX, target_testY), dim = load_target_data(target_dataset, dataset_name, train_size, test_start, test_ratio, is_synthetic)
    #target_model,_ = build_simple_mlp(n_class,dim, channel)
    
    if dataset_name == 'adult':
        target_model,_ = build_dnn(n_class,dim)
    elif dataset_name == 'purchase':
        target_model,_ = build_purchase_dnn(n_class,dim)
    elif dataset_name == 'location':
        target_model,_ = build_location_dnn(n_class,dim)
    elif dataset_name == 'avila':
        target_model,_ = build_dnn(n_class,dim)
    elif dataset_name == 'polish':
        target_model,_ = build_dnn(n_class,dim)

  
    #get_trained_keras_models(model, (target_trainX, target_trainY), (target_testX, target_testY), num_models=1)
    history = target_model.fit(target_trainX, target_trainY, epochs=epoch, batch_size=32, verbose=verbose)
    score = target_model.evaluate(target_testX, target_testY, verbose=verbose)
    _, train_acc = target_model.evaluate(target_trainX, target_trainY, verbose=verbose)
    _, test_acc = target_model.evaluate(target_testX, target_testY, verbose=verbose)
    print('\n', "Target Train acc : ", (train_acc * 100.0),"Target Test acc : ", (test_acc * 100.0))
    #print('\n', 'Model test accuracy:', score[1])
    return target_model, dim


# In[ ]:


def prepare_attack_test_data(dataset_name, attack_test_members, attack_test_nonmembers, target_model, is_synthetic):
    members = []
    nonmembers = []

    memberX, memberY, nonmemberX, nonmemberY = load_attack_test_data(dataset_name, attack_test_members, attack_test_nonmembers, is_synthetic)

    # member
    target_model_member_pred = target_model.predict(memberX, batch_size=32)
    target_model_member_class = np.argmax(memberY, axis=1)
    target_model_member_pred = np.vstack(target_model_member_pred)
    #target_model_member_class = [item for sublist in target_model_member_class for item in sublist]
    members.append(np.ones(len(target_model_member_pred)))
    members = [item for sublist in members for item in sublist]


    # nonmember
    target_model_nonmember_pred = target_model.predict(nonmemberX, batch_size=32)
    target_model_nonmember_class = np.argmax(nonmemberY, axis=1)
    target_model_nonmember_pred = np.vstack(target_model_nonmember_pred)
    #target_model_nonmember_class = [item for sublist in target_model_nonmember_class for item in sublist]
    nonmembers.append(np.zeros(len(target_model_nonmember_pred)))
    nonmembers = [item for sublist in nonmembers for item in sublist]

    full_attack_test_pred_val = (target_model_member_pred, target_model_nonmember_pred)
    full_attack_test_mem_status = (np.array(members).astype('int32'),np.array(nonmembers).astype('int32'))
    full_attack_test_class_status = (np.array(target_model_member_class),np.array(target_model_nonmember_class))

    #print('\n pred', full_attack_test_pred_val)
    #print('\n class', full_attack_test_class_status)
    #print('\n mem status', full_attack_test_mem_status)

    attack_test_data = (full_attack_test_pred_val, full_attack_test_mem_status,full_attack_test_class_status)
    
    return attack_test_data

