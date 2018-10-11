# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:57:33 2018

@author: cogillsb
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import signal
import settings
from scipy.signal import hilbert


def smooth(x,window_len=11,window='hanning'):
    '''
    Mehtod to  smooth out curves
    '''

    if x.ndim != 1:
        raise ValueError ("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError ("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y

def smooth_curves(curves):
    '''
    Loop for smooth method
    '''
    for i in range(curves.shape[0]):
        curves[i]= smooth(curves[i])[:-10]
    return curves
    

def detrend_array(curves):
    '''
    Removing baseline patterns
    '''
    for i in range(curves.shape[0]):
        curves[i]=signal.detrend(curves[i])
        
    return curves

def norm_curve(curve_array):
    '''
    Putting all of the normalization steps together. 
    '''
    curve_array = curve_array[:,settings.curve_start:]
    curve_array = detrend_array(curve_array)
    minmax_scale = MinMaxScaler(feature_range=(-1, 1)).fit(curve_array.T)
    curve_array=minmax_scale.transform(curve_array.T).T
    
    return curve_array

def noise_curve(curves):
    '''
    Move the curves around for robustness
    '''
    ht_ns = settings.height_noise
    wd_ns = settings.width_noise
    
    for i in range(curves.shape[0]):
        #alter height
        curves[i] = curves[i] + np.random.normal(-ht_ns, ht_ns)        
        #alter_period
        curves[i] = shft(curves[i], np.random.randint(-wd_ns, wd_ns))    
    return curves

def shft(curve, shft):
    '''
    Shift an array left or right while maintaining length
    '''
    if shft>0:
        curve = np.concatenate((curve[shft:],np.repeat(curve[-1],shft)))
    if shft<0:
        curve = np.concatenate((np.repeat(curve[0],abs(shft)), curve[:shft]))
    
    return curve

def calc_distance(target,test):
    '''
    Distance between vectors
    '''     
    target, test = get_shift(target, test)
    target_h = np.imag(hilbert(target))
    test_h = np.imag(hilbert(test))
    #convert to numpy arrays
    target = np.array(test)
    test = np.array(test)
    target_h = np.array(target_h)
    test_h = np.array(test_h)
    return (np.linalg.norm(target-test)
        + np.linalg.norm(target_h-test_h))


def get_shift(target, test):
    '''
    Correlate peaks for comparison
    '''
    shift = np.argmax(signal.correlate(target, test)) - (len(test)-1)
    if shift>0:
        target = target[shift:]
        test = test[:-shift]
    if shift<0:
        target = target[:shift]
        test = test[-shift:]
                
    return target, test

