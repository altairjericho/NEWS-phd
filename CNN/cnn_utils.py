import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, re, gc
import h5py
import keras
from keras.models import Model
from keras.layers import Input
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score


def precise(a_true, a_pred, thresh = 0.5):
    tp = 1e-6
    tpfp = 1e-6
    for (a_t,a_p) in zip(a_true, a_pred):
        if a_p>thresh:
            tpfp += 1
            if a_t:
                tp += 1
    if tpfp<1: print('tpfp<1, thresh:',thresh)
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

def prec_rec_curve(y_true, y_pred, thresh_curve):
    prec_curve = []
    rec_curve = []
    for th in thresh_curve:
        prec_curve.append(precise(y_true, y_pred, th))
        rec_curve.append(recalling(y_true, y_pred, th))
    return prec_curve, rec_curve


class call_roc_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_aucs = []
        #self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        #self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        scoroc = roc_auc_score(self.validation_data[1], y_pred)
        self.val_aucs.append(scoroc)
        #print('\n',epoch,'\troc_auc:',scoroc,'\n')
        return
    
def new_input_shape(model, input_shape):    
    newInput = Input(shape=input_shape)
    newOutputs = model(newInput)
    new_model = Model(newInput, newOutputs, name=model.name+'inp_chng')
    new_model.compile(optimizer=model.optimizer, loss=model.loss)
    #new_model.summary()
    return new_model
    
    
def load_data(sig_n='C100keV', bckg_n='gamma', training=True, images=True, feat_study=False, path_h5="/home/scanner-ml/Artem/Python/NEWS/data/", shuf_ind={}):
    res=[]
    if feat_study: path_h5 += 'ft_'
    with h5py.File(path_h5+'dataset.h5','r') as datafile:
        if images and feat_study:
            sig_n += '/images'
            bckg_n += '/images'
        elif feat_study:
            sig_n += '/features'
            bckg_n += '/features'
        if training:
            X_train = np.vstack((datafile[sig_n+'/train'][...], datafile[bckg_n+'/train'][...]))
            y_train = np.append(np.ones(datafile[sig_n+'/train'].shape[0]),np.zeros(datafile[bckg_n+'/train'].shape[0]))    
        X_test = np.vstack((datafile[sig_n+'/test'][...], datafile[bckg_n+'/test'][...]))
        y_test = np.append(np.ones(datafile[sig_n+'/test'].shape[0]),np.zeros(datafile[bckg_n+'/test'].shape[0]))
    gc.collect()
    if training:
        if images: X_train.resize((*X_train.shape,1))
        if not 'train' in shuf_ind.keys():
            shuf_ind['train'] = list(np.arange(X_train.shape[0]))
            np.random.shuffle(shuf_ind['train'])
        X_train = X_train[shuf_ind['train']]
        y_train = y_train[shuf_ind['train']]
        print ("X_train shape:\t" + str(X_train.shape))
        print ("y_train shape:\t" + str(y_train.shape))
        res += [X_train, y_train]
    if images: X_test.resize((*X_test.shape,1))
    if not 'test' in shuf_ind.keys():
        shuf_ind['test'] = list(np.arange(X_test.shape[0]))
        np.random.shuffle(shuf_ind['test'])
    X_test = X_test[shuf_ind['test']]
    y_test = y_test[shuf_ind['test']]
    print ("X_test shape:\t" + str(X_test.shape))
    print ("y_test shape:\t" + str(y_test.shape))
    res += [X_test, y_test, shuf_ind]
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



def load_outputs(train_l = True, val_l = True, roc_l = True, path_out="/home/scanner-ml/Artem/Python/NEWS/CNN/outputs/", epochs=['10','50'], model_spec='conv4_3d_res/v1_'):
    loss, valoss, rocs = {},{},{}
    for e in epochs:
        if train_l: loss[e] = np.loadtxt(path_out+model_spec+'loss_train_'+e+'.txt')
        if val_l: valoss[e] = np.loadtxt(path_out+model_spec+'loss_val_'+e+'.txt')
        if roc_l: rocs[e] = np.loadtxt(path_out+model_spec+'roc-auc_'+e+'.txt')
    return loss, valoss, rocs



def clean_quantile_feat(params, quant_up, quant_down, clean_key):
    """
    Dropping few percent most outlying samples using quantiles, which are bigger than quant_up or smaller than quant_down.
    """
    for key in clean_key:
        params = params[ params[key]<params[key].quantile(quant_up) ]
        params = params[ params[key]>params[key].quantile(1-quant_down) ]
    return params.dropna()



def load_data_v2(classes={'s_C30keV':'full','b_gamma':'full'}, n_folds=1, tr_val_test=[True,True,False], im_ft=[True,False], path_h5="/home/scanner-ml/Artem/Python/NEWS/data/dataset_clean.h5", shuf_ind={}, verbose=1, ddd=True, stratify=False):
    '''
    Documentation-shmocumentation
    Pseudo-cross-validation
    '''
    X,y,bound,start,end = {},{},{},{},{}
    if len(tr_val_test)<3 or len(im_ft)<2:
        print('len tr_val_test:',len(tr_val_test),', im_ft:',len(im_ft))
        return False
    
    for k in classes.keys():
        # defining which classes to load and which part of data (not to load)
        # bounds: [f, e] -> N*e//f-N//f : N*e//f
        if k[0]=='s':
            sig_n = k[2:]
            if classes[k]=='full' or n_folds==1:
                bound['s'] = [1, 0]
            else:
                bound['s'] = [n_folds, int(classes[k])]
        elif k[0]=='b':
            bckg_n = k[2:]
            if classes[k]=='full' or n_folds==1:
                bound['b'] = [1, 0]
            else:
                bound['b'] = [n_folds, int(classes[k])]
            
    with h5py.File(path_h5,'r') as datafile:
        for l in im_ft[0]*['/images/']+im_ft[1]*['/features/']:
            sig_l = sig_n+l
            bckg_l = bckg_n+l
            for t in tr_val_test[0]*['train']+tr_val_test[1]*['val']+tr_val_test[2]*['test']:
                sig_t = sig_l+t
                bckg_t = bckg_l+t
                N_s, N_b = datafile[sig_t].shape[0], datafile[bckg_t].shape[0]
                load_s, load_b = np.zeros(N_s, dtype=bool), np.zeros(N_b, dtype=bool)
                if stratify: N_str = min(N_s, N_b); N_s, N_b = N_str, N_str
                load_s[:N_s] = True; load_b[:N_b] = True
                if t=='train':
                    end["s"], end['b'] = N_s*bound['s'][1]//bound['s'][0], N_b*bound['b'][1]//bound['b'][0]
                    start['s'], start['b'] = end['s']-N_s//bound['s'][0], end['b']-N_b//bound['b'][0]
                    load_s[start['s']:end['s']] = False
                    load_b[start['b']:end['b']] = False
                X_s, X_b = datafile[sig_t][...], datafile[bckg_t][...]
                X[l+t] = np.vstack( (X_s[load_s], X_b[load_b]) )
                y[l+t] = np.append(np.ones(load_s.sum()),np.zeros(load_b.sum()))
                gc.collect()
                if l=='/images/' and ddd: X[l+t].resize((*X[l+t].shape,1))
                if not t in shuf_ind.keys():
                    shuf_ind[t] = list(np.arange(X[l+t].shape[0]))
                    np.random.shuffle(shuf_ind[t])
                X[l+t] = X[l+t][shuf_ind[t]]
                y[l+t] = y[l+t][shuf_ind[t]]
                if verbose==1:
                    print('Number of '+t+' '+sig_n+' samples: \t',X_s[load_s].shape[0])
                    print('Number of '+bckg_n+' samples: \t',X_b[load_b].shape[0])
                if verbose==2:
                    print('Number of'+' '.join((l+t).split('/'))+' samples: ',X[l+t].shape[0])
                gc.collect()
    return X,y,shuf_ind