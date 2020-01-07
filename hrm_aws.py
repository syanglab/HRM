# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:13:19 2019

@author: cogillsb
"""

import os
from keras.layers import Input, Dense, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import sqlite3
import io


def output_plt(seqs, curves, ids, generator):
    '''
    plot out the data
    '''
    for i in range(0,len(curves)):
    
        of = '%s_v3_curves.pdf' % ids[i]
        
        #Plot the actual curve
        plt.plot(np.arange(80.1,95.0,0.1), curves[i], label='Experimental',color='red')

        #Plot the pred_curve
        gen_images = generator.predict([np.array([seqs[i]])])
        plt.plot(np.arange(80.1,95.0,0.1), gen_images[0], label='Pred',color='blue')        
        
        #Show the plot
        plt.ylabel('Dissociation')
        plt.legend(loc="upper right")
        plt.xlabel('Temperature (C)')
        plt.ylabel('Dissociation')    
        plt.tight_layout()
        plt.title('%s' % ids[i][:-1])
        plt.savefig(of)
        
        #Export
        os.system("aws s3 cp %s s3://bucketname/%s" % (of,of))

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def build_generator(seq_len):
    seq_input = Input(shape = (seq_len,))
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

    
def build_discriminator(curve_len, seq_len):
    curve_input = Input(shape = (curve_len,))
    seq_input = Input(shape = (seq_len,))
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

def build_gan(discriminator, generator, seq_len):
    
    discriminator.trainable=False
    
    gan_input = Input(shape=(seq_len,))
    
    x = generator(gan_input)
    
    gan_output= discriminator([x, gan_input])
    
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    
    return gan

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


#Load the data
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
conn = sqlite3.connect('HRM_syang.sqlite', detect_types=sqlite3.PARSE_DECLTYPES)
cur = conn.cursor()
#Convert the sequences to numpy vectors
base_enc = {
    'A':0.25,
    'T':0.5,
    'C':0.75,
    'G':1.0,
    'N':(0.25 + 0.5 + 0.75 + 1.00)/4,
    'M':(0.75 + 0.25)/2,
    'Y':(0.5 + 0.75)/2,
    'X':(0.25 + 0.5 + 0.75 + 1.00)/4,
    '-':0.0,
    'S':(1.0 + 0.75)/2  
}

sql_command = """
SELECT raw_sequences.org_id, aligned_sequences.sequence, curves.norm_curve
FROM curves
INNER JOIN raw_sequences ON aligned_sequences.seq_id=raw_sequences.id
INNER JOIN aligned_sequences ON curves.seq_id=aligned_sequences.seq_id;
"""
curve_mat = []
seq_mat = []
tag_mat = []
cur.execute(sql_command)
data = cur.fetchall()
for d in data:
    curve_mat.append(d[2])
    seq_mat.append([base_enc[base] for base in d[1]])
    tag_mat.append(d[0])
curves = np.array(curve_mat)/100
seqs = np.array(seq_mat)
ids = np.array(tag_mat)

curve_len = curves.shape[1]
seq_len = seqs.shape[1]

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
    generator= build_generator(seq_len)
    discriminator= build_discriminator(curve_len, seq_len)
    gan = build_gan(discriminator, generator, seq_len)
    

    for e in range(50000):
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
    of = '%s_v3_pred_curve.npy' % ids_test[0]
    np.save(of, gen_images_test)
    
    #Export
    os.system("aws s3 cp %s s3://bucketname/%s" % (of,of))