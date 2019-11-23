# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:13:19 2019

@author: cogillsb
"""

import os
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate, Flatten, dot
from keras.layers import Conv1D, UpSampling1D, Activation, Lambda, TimeDistributed, Bidirectional, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
import keras.backend as K
import numpy as np

from keras.layers.merge import _Merge
from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import KFold

def output_plt(seqs, curves, ids, generator):
    '''
    plot out the data
    '''
    for i in range(0,len(curves)):
    
        of = '%s_curves.pdf' % ids[i][:-1]
        
        #Plot the actual curve
        plt.plot(np.arange(80.1,95.0,0.1), curves[i], label='Experimental',color='red')

        #Plot the pred_curve
        gen_images = generator.predict([np.array([seqs[i]])])
        plt.plot(np.arange(80.1,95.0,0.1), gen_images[0], label='Pred',color='blue')
        
        #Plot the umelt curve
        crv = np.load('%s_umelt_mat.npy' % ids[i][:-1])
        plt.plot(np.arange(80.1,95.0,0.1), crv, label='UMelt',color='green')
        
        #Show the plot
        plt.ylabel('Dissociation')
        plt.legend(loc="upper right")
        plt.xlabel('Temperature (C)')
        plt.ylabel('Dissociation')    
        plt.tight_layout()
        plt.title('%s' % ids[i][:-1])
        plt.savefig(of)
        
        #Export
        os.system("aws s3 cp %s s3://dxoracle1/hrm/%s" % (of,of))

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def build_generator():
    seq_input = Input(shape = (214,))
    first = Dense(256)(seq_input)
    first = LeakyReLU(0.2)(first)
    
    second = Dense(512)(first)
    second = LeakyReLU(0.2)(second)
    
    third = Dense(1028)(second)
    third = LeakyReLU(.2)(third)
    
    out = Dense(150)(third)
    out = LeakyReLU(.2)(out)
    
    gen = Model([seq_input,], out)
    gen.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    
    return gen

    
def build_discriminator():
    curve_input = Input(shape = (150,))
    seq_input = Input(shape = (214,))
    dis_input = concatenate([curve_input, seq_input])
    first = Dense(1028)(dis_input)
    first = LeakyReLU(0.2)(first)
    
    second = Dense(512)(first)
    second = LeakyReLU(0.2)(second)
    
    third = Dense(256)(second)
    third = LeakyReLU(.2)(third)
    
    out = Dense(1, activation='sigmoid')(third)
    
    dis = Model([curve_input, seq_input,], out)
    
    dis.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    
    return dis

def build_gan(discriminator, generator):
    
    discriminator.trainable=False
    
    gan_input = Input(shape=(214,))
    
    x = generator(gan_input)
    
    gan_output= discriminator([x, gan_input])
    
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    
    return gan

#Load the data
seqs = np.load('seq_mat_half.npy')
curves = np.load('curve_mat_half.npy')
ids = np.load('id_mat_half.npy')

#Split the data for a loocv
kf = KFold(n_splits=len(curves))

#Run each fold
for train_index, test_index in kf.split(seqs):
    
    #Split the matrices
    seqs_train, seqs_test = seqs[train_index], seqs[test_index]
    curves_train, curves_test = curves[train_index], curves[test_index]
    ids_train, ids_test = ids[train_index], ids[test_index]
    
    print('Testing %s' % str(ids_test[0]))
    
    # Creating GAN
    generator= build_generator()
    discriminator= build_discriminator()
    gan = build_gan(discriminator, generator)
    

    for e in range(200):
        #Predict curves
        gen_images = generator.predict([seqs_train])

        #Train the discriminator
        #Label real curves
        y_rl = np.random.uniform(low=0.7, high=1.0, size=(len(seqs_train)))
        

        #Turn on the discriminator
        discriminator.trainable = True

        #Train on real curves
        dloss = discriminator.train_on_batch([curves_train, seqs_train], y_rl)

        # Label fake curves
        y_fk = np.random.uniform(low=0.0, high=0.3, size=(len(seqs_train)))

        #Train on fake curves
        dloss = discriminator.train_on_batch([gen_images,  seqs_train], y_fk)
                                              

        #Train the generator
        #Turn off the discriminator
        discriminator.trainable = False

        #Train on fake curves
        gloss = gan.train_on_batch([seqs_train], y_rl)

    #plt results
    output_plt(seqs_test, curves_test, ids_test, generator) 
    
    #Outut the matrices
    gen_images_test = generator.predict([seqs_test])
    of = '%s_pred_curve.npy' % ids_test[0][:-1]
    np.save(of, gen_images_test)
    
    #Export
    os.system("aws s3 cp %s s3://dxoracle1/hrm/%s" % (of,of))