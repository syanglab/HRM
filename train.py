# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:29:13 2018

@author: cogillsb
"""
from model import GANModel
import numpy as np
import settings 
import curve_manip

class Train(object):
    def __init__(self, seqs, curves, tags):
        self.random_dim = settings.random_dim
        self.curves = curves
        self.seqs = seqs
        self.tags = tags
        self.model = GANModel(self.curves.shape[1], self.seqs.shape[1])
    
    def train_gan(self, e):
        '''
        Run an epoch for model.
        '''
        flip, odd = self.check_swicthes(e)
        #Train discriminator
        self.model.discriminator.trainable = True
        dloss = self.model.discriminator.train_on_batch(
                self.generate_images(),
                self.generate_disc_labels(flip))
        #Train generator
        self.model.discriminator.trainable = False
        gloss = self.model.gan.train_on_batch(
                self.sample_dat(),
                self.generate_gen_labels(odd))        

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

    def generate_images(self):
        '''
        Run generator to produce images with correct and incorrect labels.
        '''
        noise = np.random.normal(0,1, size = [settings.batch_size, self.random_dim])
        idx = np.random.randint(0, self.curves.shape[0], settings.batch_size)
        image_batch = curve_manip.noise_curve(self.curves[idx])
        seqs_batch = self.seqs[idx]
        #Get mis match seqs 
        seqs_batch_mis = self.seqs[self.get_wrong_seqs(idx)]
        
        generated_images = self.model.generator.predict([noise,  seqs_batch])
        X = np.concatenate([image_batch, image_batch, generated_images])
        seqs_batch = np.concatenate([seqs_batch, seqs_batch_mis, seqs_batch]) 
               
        return [X, seqs_batch]
    
    def sample_dat(self):
        '''
        Randomly sampling seq and noise data
        '''
        noise = np.random.normal(0, 1, size=[settings.batch_size, self.random_dim])
        idx = np.random.randint(0, self.curves.shape[0], settings.batch_size)
        seqs_batch = self.seqs[idx]
        
        return [noise, seqs_batch]
    
    
    def generate_disc_labels(self, flip):
        '''
        Smoothed discriminator labels. Correct unless flipped. 
        '''
        # Labels for generated and real data
        y_dis = np.zeros(3*settings.batch_size)
        dbl_bs = 2*settings.batch_size
        
        if flip:
            #Smoothing
            y_dis[settings.batch_size:]=np.array(np.random.randint(700, 1200, dbl_bs)/1000)
            y_dis[:settings.batch_size]=np.array(np.random.randint(0, 300, settings.batch_size)/1000)
        else:
            #Smoothing
            y_dis[settings.batch_size:]=np.array(np.random.randint(0, 300, dbl_bs)/1000)
            y_dis[:settings.batch_size]=np.array(np.random.randint(700, 1200, settings.batch_size)/1000)
        #Flip after a while
        if flip:
            y_dis=np.flip(y_dis, axis=0)
        
        return y_dis
    
    def generate_gen_labels(self, odd):
        '''
        Alternating between one and zero labels
        '''
        if odd:
            return np.zeros(settings.batch_size)
        else:
            return np.ones(settings.batch_size)
    
   
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
    
     
        
    
