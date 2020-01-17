# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:17:19 2020

@author: cogillsb
"""

from scipy.signal import hilbert
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn import metrics
from scipy import signal
import itertools


def format_raw_hrm(raw_curve_fn, org_list_fn):
    '''The data coming from the instrument is in an awkward
    format. This takes in the txt format, labels it with a 
    organism tag list, and outputs the df.'''
    
    #Read in the raw data
    df = pd.read_csv(raw_curve_fn, sep ='\t')
    df.drop(columns='Text', inplace=True)
    c_d = [x for x in df.columns if "X" in x and len(x)>1]
    df.drop(columns=c_d, inplace=True)
    
    #Label the data
    o_lst = pd.read_csv(org_list_fn)
    cols = list(o_lst.orgs.values)
    cols.insert(0, 'X')
    df.columns= cols
    
    #Round out the temps
    X = df.X.round(1).values
    
    #Transpose the dataframe       
    df.drop(columns=['X'], inplace=True)
    df = df.T
    df.columns = X        
    df.index.name  = 'org_names'
    df.reset_index(inplace=True)
    
    return df
    

def norm_curve(curve, temps, norm_len):
    '''Normalizing a curve using the
    linear correction method outlined
    by the umelt group.'''
    
    def get_slope(temp_1, temp_2, fluor_1, fluor_2):
            slope = (fluor_2-fluor_1)/(temp_2-temp_1)
            return slope
    def getIntercept(x, y, m):
        b = y-(x*m)
        return b
    def getPoint(m, x, b):
        point = m*x+b
        return point
    
    new_curve = []
    top_strt = 0
    top_stp = norm_len
    bot_strt = -(norm_len + 1)
    bot_stp = len(curve)-1
    m1 = get_slope(temps[top_strt], temps[top_stp], curve[top_strt], curve[top_stp])
    m2 = get_slope(temps[bot_strt], temps[bot_stp], curve[bot_strt], curve[bot_stp])
    b1 = getIntercept(temps[top_stp], curve[top_stp], m1)
    b2 = getIntercept(temps[bot_strt], curve[bot_strt], m2)
    
    for i in range(len(curve)):
        topF = getPoint(m1, temps[i], b1)
        botF = getPoint(m2, temps[i], b2)
        newF = ((curve[i]-botF)/(topF-botF))
        new_curve.append(newF)

    rng = max(new_curve)-min(new_curve)
    new_curve = [((x-min(new_curve))/rng) * 100 for x in new_curve]
    for i in range(new_curve.index(max(new_curve))):
        new_curve[i] = max(new_curve)
    for i in range(new_curve.index(min(new_curve)), len(new_curve)):
        new_curve[i] = min(new_curve) 
    
    return new_curve




def calc_distance(target,test):
    '''
    Distance between vectors
    ''' 
    def equalize(target,test):
        d=len(target)-len(test)
        if d>0:
            target=target[:-d]
        if d<0:
            d = abs(d)
            test=test[:-d]
        return target, test
    
    def get_shift(ys):
        ks=list(ys.keys())    
        for tk in ks:
            if tk == ks[0]:
                continue
            shift = np.argmax(signal.correlate(ys[ks[0]], ys[tk])) - (len(ys[tk])-1)        
            if shift>0:            
                #Shift test to the right
                for k in ks:
                    if k != tk:
                        ys[k]=ys[k][shift:]
                ys[tk]=ys[tk][:-shift]
            if shift<0:
                shift=abs(shift)
                #Shift test to the left
                for k in ks:
                    if k != tk:
                        ys[k]=ys[k][:-shift]
                ys[tk]=ys[tk][shift:]

        return ys
    
    def norm(ys):
        for key, val in ys.items():
            val = np.array(val)
            ys[key] = (val - val.min()) / (val.max() - val.min())
        return ys

    #target,test = equalize(target,test)
    ys = {'target':target,'test':test}
    #ys = norm(ys)
    #ys = get_shift(ys)
    
    target=ys['target']
    test=ys['test']
    #target,test = new_shift(target,test)
    target = signal.detrend(target)
    test = signal.detrend(test)
    target_h = np.imag(hilbert(target))
    test_h = np.imag(hilbert(test))
    #convert to numpy arrays
    target = np.array(test)
    test = np.array(test)
    target_h = np.array(target_h)
    test_h = np.array(test_h)
    
    dist = np.linalg.norm(target-test) + np.linalg.norm(target_h-test_h)
    
    return dist

def smooth(x,window_len=11,window='hanning'):
   
    l = len(x)
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
    
    cut = len(y)-l
    y = y[int(cut/2):-round(cut/2)]

            
    return y






        

        
        
        
            