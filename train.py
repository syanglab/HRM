# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:29:13 2018

@author: cogillsb
"""
from model_ae import GANModel
import numpy as np
import settings 
import random

class Train(object):
    def __init__(self, seqs, curves, tags):
        self.random_dim = settings.random_dim
        self.curves = curves
        self.seqs = seqs
        self.tags = tags
        self.model = GANModel(self.curves.shape[1], self.seqs.shape[1])
        self.idx = list(range(curves.shape[0]))
        self.gloss = 0
        self.dloss = 0
    def train_gan(self, e):
        '''
        Run an epoch for model.
        '''
        
        flip, odd = self.check_swicthes(e)
        flip = False
        odd = True
        #train discriminator
        self.model.discriminator.trainable = True
        #for i in range(5):
        random.shuffle(self.idx)
        for i in range(1, int(self.curves.shape[0]/settings.batch_size)+1):                    
            #Get a batch
            start = i*settings.batch_size-settings.batch_size
            stop = i*settings.batch_size
            idx = self.idx[start:stop]
            #Get disc data
            labels = self.seqs[idx]
            generated_images = self.model.generator.predict([labels])
            X = np.concatenate([self.curves[idx], self.curves[idx], generated_images])
            seqs_batch = np.concatenate([labels, self.seqs[self.get_wrong_seqs(idx)], labels]) 
            y_dis = np.zeros(3*settings.batch_size)
            dbl_bs = 2*settings.batch_size
            if flip:
                #Swap discriminator labels
                y_dis[settings.batch_size:]=np.array(np.random.randint(700, 1200, dbl_bs)/1000)
                y_dis[:settings.batch_size]=np.array(np.random.randint(0, 300, settings.batch_size)/1000)
                           
            else:
                y_dis[settings.batch_size:]=np.array(np.random.randint(0, 300, dbl_bs)/1000)
                y_dis[:settings.batch_size]=np.array(np.random.randint(700, 1200, settings.batch_size)/1000)
            
            dloss = self.model.discriminator.train_on_batch([X,seqs_batch], y_dis)        
            
            self.dloss = dloss
        
        #Train generator
        self.model.discriminator.trainable = False
        random.shuffle(self.idx)
        for i in range(1, int(self.curves.shape[0]/settings.batch_size)+1):
            start = i*settings.batch_size-settings.batch_size
            stop = i*settings.batch_size
            idx = self.idx[start:stop]
            if odd:
                gloss = self.model.gan.train_on_batch(self.seqs[idx], np.ones(settings.batch_size))
            else:
                gloss = self.model.gan.train_on_batch(self.seqs[idx], np.zeros(settings.batch_size))
            self.gloss=gloss
            
        #Output quick performance check
        if e%settings.chkpt == 0:
            print(e)
            print("Gloss")
            print(gloss)
            print("Dloss")
            print(dloss)
            

    def get_wrong_seqs(self, idx):
        '''
        Get indces for sequences that are
        different from the correct sequence
        '''
        #Get mis match seqs 
        wrong_seqs = []
        for i in idx:
            matching = True
            while matching:
                j = np.random.randint(0, self.curves.shape[0])
                if not np.array_equal(self.seqs[i], self.seqs[j]):
                    matching = False
                    wrong_seqs.append(j)
        
        return wrong_seqs

    def sample_dat(self, idx):
        '''
        Randomly sampling seq and noise data
        '''
        noise = np.random.normal(0, 1, size=[settings.batch_size, self.random_dim])
        #idx = np.random.randint(0, self.curves.shape[0], settings.batch_size)
        seqs_batch = self.seqs[idx]
        
        return [noise, seqs_batch]
    
    def check_swicthes(self, e):
        '''
        Check points for challenges to network. 
        '''
        if e%50 ==0 and e%settings.chkpt != 0:
            flip = True
        else:
            flip = False
            
        if e%2 != 0:
            odd = True
        else:
            odd = False            
           
        return flip, odd
    

   

     
        
    
