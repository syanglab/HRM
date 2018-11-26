# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:54:43 2018

@author: cogillsb
"""
from preproc import PreProc
from train import Train
from test import Test
import random
import settings
import pickle
from ppwseqs import PreProcwSeqs

def split_orgs(tags):
    '''
    Getting the test and train indices by organism
    '''
    test_ind = []
    train_ind = []
    #Get the org names
    orgs = list(set([x.split('_')[0] for x in tags]))
    #Shuffle with repeatiblity and split out percent
    orgs.sort()
    random.Random(settings.seed).shuffle(orgs)
    test_orgs = orgs[:round(len(orgs)*settings.test_percent)] 
    #assign indices
    for i in range(len(tags)):
        if tags[i].split('_')[0] in test_orgs:
            test_ind.append(i)
        else:
            train_ind.append(i)
    
    return test_ind, train_ind
            
            
def main():
    '''
    Run the program
    '''
    filename = '/my_data/dat_16s_glblalgn_parse_curve_norm.pkl'
    filehandler = open(filename, 'rb')
    dat = pickle.load(filehandler)
    
    test_ind, train_ind = split_orgs(dat.tags)
    train = Train(dat.seqs[train_ind],
            dat.curves[train_ind],
            dat.tags[train_ind])
    
    test = Test(dat.curves[test_ind],
                dat.seqs[test_ind],
                dat.tags[test_ind],
                dat.curves[train_ind],
                dat.seqs[train_ind],
                dat.tags[train_ind])
    
    for e in range(settings.epochs):
        
        train.train_gan(e)
        if e%settings.chkpt == 0:
            #Save the model
            train.model.generator.save('/output/gen_model_%i.h5' % e)
            train.model.discriminator.save('/output/disc_model_%i.h5' % e)
            
            test.test_model(train.model.generator, train.model.discriminator, e)

if __name__ == "__main__":
    main()





