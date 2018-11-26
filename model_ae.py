# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:53:56 2018

@author: cogillsb
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Flatten, Reshape
from keras.layers import Conv1D, UpSampling1D, Activation, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import settings
import keras.backend as K
import numpy as np
from functools import partial
from keras.models import load_model

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
        #self.generator = load_model('/my_data/gen_model_29000.h5')       
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.adam)
        #self.discriminator = load_model('/my_data/disc_model_29000.h5')
        self.discriminator=self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.gan = self.build_gan()
        self.gan.compile(loss='binary_crossentropy', optimizer=self.adam)
        
        
    def build_generator(self):
        '''
        Generate a melt curve
        '''
        
        """
        Creates the generator according to the specs in the paper below.
        [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
        :param model:
        :return:
        """

        
        filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]
        label_in = Input(shape=(self.seq_len,))
        encoder = Lambda(lambda x: K.expand_dims(x, axis=2))(label_in) 
        
        for filter_size in filter_sizes:
         
            encoder = Conv1D(filter_size, 4, strides=2, border_mode='same')(encoder)
            # paper skips batch norm for first layer
            encoder = LeakyReLU(alpha=self.lk_rl)(encoder)
            if filter_size != 64:
                encoder = BatchNormalization()(encoder)
            encoder = Dropout(p=0.5)(encoder)
            

        
        filter_sizes = [512, 512, 256, 128, 64]
    
        decoder = encoder
        for filter_size in filter_sizes:
            decoder = UpSampling1D(size=(2))(decoder)
            decoder = Conv1D(filter_size, 4, border_mode='same')(decoder)
            decoder = LeakyReLU(alpha=self.lk_rl)(decoder)
            decoder = BatchNormalization()(decoder)
            decoder = Dropout(p=0.5)(decoder)
           
    

        decoder = Conv1D(1, 4, border_mode='same')(decoder)
        decoder = Flatten()(decoder)
        output = Activation('tanh')(decoder)
        
        
        return Model(inputs=[label_in], outputs=output)

        
    def build_discriminator(self):
        #Fetch params
        img_in = Input(shape=(self.curve_len,))
        label_in = Input(shape=(self.seq_len,))
        
        encoder = concatenate([img_in, label_in])
        encoder = Lambda(lambda x: K.expand_dims(x, axis=2))(img_in) 
         
        
        depths = [64, 128, 256, 512]
        
        for depth in depths:
            encoder = Conv1D(depth, 4, strides=2, padding='same')(encoder)
            encoder = LeakyReLU(alpha=self.lk_rl)(encoder)            
            encoder = BatchNormalization()(encoder)
            encoder = Dropout(p=0.5)(encoder)
    
        encoder = Flatten()(encoder)
        
        
        output = Dense(1, activation='sigmoid')(encoder)        
        
        return Model(inputs = [img_in, label_in], outputs = output,)   
        
    
    
    
    def build_gan(self):
        # Combined network
        self.discriminator.trainable = False
        label_in = Input(shape=(self.seq_len,))        
        
        img =self.generator(label_in)
        ganOutput = self.discriminator([img, label_in])
        
        return Model(inputs=[label_in], outputs=ganOutput)
    

  
        