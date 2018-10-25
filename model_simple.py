# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:53:56 2018

@author: cogillsb
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Flatten, Reshape
from keras.layers import Conv1D, Activation, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import settings
import keras.backend as K
import numpy as np
from functools import partial

from keras.layers.merge import _Merge


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((settings.batch_size, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class GANModel(object):
    
    def __init__(self, curve_len, seq_len):
        
        #Parameters
        self.drp = settings.dropout
        self.gradient_penalty_weight = 10
        self.mom = settings.momentum
        self.lk_rl = settings.leaky_relu        
        self.random_dim = settings.random_dim
        self.curve_len = curve_len
        self.seq_len = seq_len        
        self.adam = Adam(lr=0.0002, beta_1=0.5)        
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.gan = self.build_gan()
        self.gan.compile(loss=self.wasserstein_loss, optimizer=self.adam)
        self.critic = self.build_critic()
        
    
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
        
    
    
    
    def build_generator(self):
        '''
        Generate a melt curve
        '''
        #Fetch parameters
        noise_in = Input(shape=(self.random_dim,))
        label_in = Input(shape=(self.seq_len,))
        
        
        decoder = concatenate([noise_in, label_in])
        
        decoder = Dense(256)(decoder) 
        decoder = LeakyReLU(alpha=self.lk_rl)(decoder)       
        decoder = Dropout(self.drp)(decoder)
        
        decoder = Dense(512)(decoder) 
        decoder = LeakyReLU(alpha=self.lk_rl)(decoder)       
        decoder = Dropout(self.drp)(decoder)
        
        decoder = Dense(1024)(decoder) 
        decoder = LeakyReLU(alpha=self.lk_rl)(decoder)       
        output = Dropout(self.drp)(decoder)
        output = Dense(1024, activation='tanh')(decoder)       
        
        return Model(inputs=[noise_in, label_in], outputs=output)
    

    def build_discriminator(self):
        #Fetch params
        img_in = Input(shape=(self.curve_len,))
        label_in = Input(shape=(self.seq_len,))
        
        encoder = concatenate([img_in, label_in])
        
        encoder = Dense(512)(img_in) 
        encoder = LeakyReLU(alpha=self.lk_rl)(encoder)       
        encoder = Dropout(self.drp)(encoder)
        
        encoder = Dense(256)(encoder) 
        encoder = LeakyReLU(alpha=self.lk_rl)(encoder)       
        encoder = Dropout(self.drp)(encoder)
        
        encoder = Dense(128)(encoder) 
        encoder = LeakyReLU(alpha=self.lk_rl)(encoder)       
        encoder = Dropout(self.drp)(encoder)
        
        output = Dense(1)(encoder)        
        
        return Model(input = [img_in, label_in], output = output,)
        
   
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        
        return K.mean(gradient_penalty)  
    
    
    def build_gan(self):
        # Combined network
        self.discriminator.trainable = False
        noise = Input(shape=(self.random_dim,))
        label_in = Input(shape=(self.seq_len,))        
        
        img =self.generator([noise, label_in])
        ganOutput = self.discriminator([img, label_in])
        
        return Model(inputs=[noise, label_in], outputs=ganOutput)
    
    def build_critic(self):
        self.discriminator.trainable = True
         
        #Inputs
        real_samples = Input(shape=(self.curve_len,))
        gen_samples = Input(shape=(self.curve_len,))
        
        labels = Input(shape=(self.seq_len,))
        
        
        #Discriminator vals
        disc_out_gen = self.discriminator([gen_samples, labels])
        disc_out_real = self.discriminator([real_samples, labels])
        
        #Get the average
        avg_samples = RandomWeightedAverage()([real_samples, gen_samples])
        disc_out_avg = self.discriminator([avg_samples, labels])
        
        #Define the loss function        
        partial_gp_loss = partial(self.gradient_penalty_loss,
                              averaged_samples=avg_samples,
                              gradient_penalty_weight=self.gradient_penalty_weight)
        
        partial_gp_loss.__name__ = 'gradient_penalty'
        
                
        disc_model =  Model(inputs = [real_samples, gen_samples, labels],
                           outputs = [disc_out_real, disc_out_gen, disc_out_avg])

        disc_model.compile(optimizer=self.adam, loss=[self.wasserstein_loss, 
                                              self.wasserstein_loss,
                                              partial_gp_loss])
        return disc_model    
        