import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, re, gc
import h5py
import keras
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


def precise(a_true, a_pred, thresh = 0.5):
    tp = 0
    tpfp = 0
    for (a_t,a_p) in zip(a_true, a_pred):
        if a_p>thresh:
            tpfp += 1
            if a_t:
                tp += 1
    return tp/tpfp

def accurate(a_true, a_pred, thresh = 0.5):
    tp = 0
    tn = 0
    for (a_t,a_p) in zip(a_true, a_pred):
        if a_p>thresh and a_t:
            tp += 1
        if a_p<thresh and (not a_t):
            tn += 1
    return (tp+tn)/len(a_true)

def recalling(a_true, a_pred, thresh = 0.5):
    tp = 0
    tpfn = 0
    for (a_t,a_p) in zip(a_true, a_pred):
        if a_t:
            tpfn += 1
            if a_p>thresh:
                tp += 1
    return tp/tpfn


class call_roc_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_aucs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        scoroc = roc_auc_score(self.validation_data[1], y_pred)
        self.val_aucs.append(scoroc)
        print('\n',epoch,'\troc_auc:',scoroc,'\n')
        return
    
    
def load_data(sig_n='C100keV', bckg_n='gamma', training=True, path_h5="/home/scanner-ml/Artem/Python/NEWS/data/"):
    res=[]
    with h5py.File(path_h5+'dataset.h5','r') as datafile:
        if training:
            X_train = np.vstack((datafile[sig_n+'/train'][...], datafile[bckg_n+'/train'][...]))
            y_train = np.append(np.ones(datafile[sig_n+'/train'].shape[0]),np.zeros(datafile[bckg_n+'/train'].shape[0]))    
        X_test = np.vstack((datafile[sig_n+'/test'][...], datafile[bckg_n+'/test'][...]))
        y_test = np.append(np.ones(datafile[sig_n+'/test'].shape[0]),np.zeros(datafile[bckg_n+'/test'].shape[0]))
    gc.collect()
    if training:
        X_train.resize((*X_train.shape,1))
        shuf_ind = list(np.arange(X_train.shape[0]))
        np.random.shuffle(shuf_ind)
        X_train = X_train[shuf_ind]
        y_train = y_train[shuf_ind]
        print ("X_train shape:\t" + str(X_train.shape))
        print ("y_train shape:\t" + str(y_train.shape))
        res += [X_train, y_train]
    X_test.resize((*X_test.shape,1))
    shuf_ind = list(np.arange(X_test.shape[0]))
    np.random.shuffle(shuf_ind)
    X_test = X_test[shuf_ind]
    y_test = y_test[shuf_ind]
    print ("X_test shape:\t" + str(X_test.shape))
    print ("y_test shape:\t" + str(y_test.shape))
    res += [X_test, y_test]
    gc.collect()
    return res


def pos_neg(y, preds):
    pos = np.array([[],[]]).T
    neg = np.array([[],[]]).T
    preds = np.ravel(preds)
    for i,grek in enumerate(y):
        if grek:
            pos = np.vstack((pos, np.array([preds[i],i])))
        else:
            neg = np.vstack((neg, [preds[i],i]))
    gc.collect()
    return pos, neg



def load_outputs(train_l = True, val_l = True, rocs = True, path_out="/home/scanner-ml/Artem/Python/NEWS/CNN/outputs/", epochs=['10','50'], model_spec='conv4_3d_res/v1_'):
    loss, valoss, rocs = {},{},{}
    for e in epochs:
        if train_l: loss[e] = np.loadtxt(path_out+model_spec+'loss_train_'+e+'.txt')
        if val_l: valoss[e] = np.loadtxt(path_out+model_spec+'loss_val_'+e+'.txt')
        if rocs: rocs[e] = np.loadtxt(path_out+model_spec+'roc-auc_'+e+'.txt')
    return loss, valoss, rocs
