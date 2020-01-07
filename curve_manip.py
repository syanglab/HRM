# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:17:19 2020

@author: cogillsb
"""
import pandas as pd


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
        

        
        
        
            