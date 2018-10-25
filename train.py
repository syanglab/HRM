# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:29:13 2018

@author: cogillsb
"""
from model_simple import GANModel
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
        random.shuffle(list(self.idx))
        
        for i in range(1, int(self.curves.shape[0]/settings.batch_size)+1):
            start = i*settings.batch_size-settings.batch_size
            stop = i*settings.batch_size
            idx = self.idx[start:stop]            
            widx = self.get_wrong_seqs(idx)

            #Train discriminator
            self.model.discriminator.trainable = True
            noise = np.random.rand(settings.batch_size, self.random_dim)
            labels = self.seqs[idx]
            gen_imgs = self.model.generator.predict([noise,  labels])
            dloss = self.model.critic.train_on_batch(
                    [self.curves[idx], gen_imgs, labels],
                    [np.ones(settings.batch_size), -np.ones(settings.batch_size), np.zeros(settings.batch_size)])
            
            #Train discriminator with wrong seqs
            labels = self.seqs[widx]
            dloss = self.model.critic.train_on_batch(
                    [self.curves[idx], self.curves[idx], labels],
                    [np.ones(settings.batch_size), -np.ones(settings.batch_size), np.zeros(settings.batch_size)])
            
            
            self.dloss = dloss
            #Train generator
            self.model.discriminator.trainable = False
            gloss = self.model.gan.train_on_batch(
                    self.sample_dat(idx),
                    np.ones(settings.batch_size))        
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
    
    

   

     
        
    
