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
from keras.layers import Conv2DTranspose





class GANModel(object):
    
    def __init__(self, curve_len, seq_len):
        
        #Parameters
        self.drp = settings.dropout
        self.mom = settings.momentum
        self.lk_rl = settings.leaky_relu        
        self.random_dim = settings.random_dim
        self.curve_len = curve_len
        self.seq_len = seq_len        
        self.adam = Adam(lr=0.0002, beta_1=0.5)        
        self.generator = self.build_generator()        
        self.discriminator=self.build_discriminator()
        

              

    def Conv1DTranspose(self, input_tensor, filters, kernel_size, strides=2, padding='same'):
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x

    
    def conv_labels(self, input_tensor):
        depth = 64
        labels = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)  
        for i in range(6):
            labels = Conv1D(depth, 4, strides=2, padding='same')(labels)
            labels = BatchNormalization(momentum=self.mom)(labels)
            labels = LeakyReLU(alpha=self.lk_rl)(labels)       
        
        labels = Flatten()(labels)        
        labels = Dense(512, activation='relu')(labels)
        labels = Dropout(self.drp)(labels)
        labels = Dense(256, activation='relu')(labels)
        labels = Dropout(self.drp)(labels)
        labels = Dense(128, activation ='relu')(labels)
            
        return labels
        
    def build_generator(self):
        '''
        Generate a melt curve
        '''
        #Fetch parameters
        noise_in = Input(shape=(self.random_dim,))
        label_in = Input(shape=(self.seq_len,))
        label = self.conv_labels(label_in)
        decoder = concatenate([label,noise_in])
        
        #Transposed 1d convolution
        dim = 128
        depth = 64
        decoder = Dense(dim*depth)(decoder)
        decoder = Reshape((dim,depth))(decoder)
        decoder = BatchNormalization(momentum=self.mom)(decoder)
        decoder = LeakyReLU(self.lk_rl)(decoder)
             
        #256
        decoder = self.Conv1DTranspose(decoder, depth, 5, padding='same')
        decoder = BatchNormalization(momentum=self.mom)(decoder)
        decoder = LeakyReLU(self.lk_rl)(decoder)

        #512
        decoder = self.Conv1DTranspose(decoder, depth, 5, padding='same')
        decoder = BatchNormalization(momentum=self.mom)(decoder)
        decoder = LeakyReLU(self.lk_rl)(decoder)

        #1024
        decoder = self.Conv1DTranspose(decoder, 1, 5, padding='same')
        decoder = Activation('tanh')(decoder)
        
        output = Lambda(lambda x: K.squeeze(x, axis=2))(decoder)
        
        return Model(inputs=[noise_in, label_in], outputs=output)
    

    def build_discriminator(self):
        #Fetch params
        img_in = Input(shape=(self.curve_len,))
        label_in = Input(shape=(self.seq_len,))
        label = self.conv_labels(label_in)
        encoder = Lambda(lambda x: K.expand_dims(x, axis=2))(img_in) 
        depth = 64
        
        for i in range(4):
            encoder = Conv1D(depth, 5, strides=2, padding='same', kernel_initializer='he_normal')(encoder)
            #encoder = BatchNormalization(momentum=self.mom)(encoder)
            encoder = LeakyReLU(alpha=self.lk_rl)(encoder)
    
        encoder = Flatten()(encoder)
        encoder = concatenate([encoder, label])        
        encoder = Dense(512, activation='relu', kernel_initializer='he_normal')(encoder)
        encoder = LeakyReLU(alpha=self.lk_rl)(encoder)
        #encoder = Dropout(self.drp)(encoder)  
        
        output = Dense(1, kernel_initializer='he_normal')(encoder)        
        
        return Model(inputs = [img_in, label_in], outputs = output,)
        
   
    
    
    

    
