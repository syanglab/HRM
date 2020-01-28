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
import sys
sys.path.insert(0, './HRM')
import curve_manip as cm

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    seq = ''
    for b in dna[::-1]:
        if b in complement:
            seq += complement[b]
        else:
            seq += b
    return seq



def output_plt(seqs, rev_seqs, umelt, curves, ids, generator, tag):
    '''
    plot out the data
    '''
    ind = 0
    plt.rcParams['figure.figsize'] = [15,110]
    of = '%s_v3_curves.pdf' % tag
    for i in range(0,len(curves)):
        ind += 1
        plt.subplot(30, 2,ind)       
                
        #Plot the actual curve
        plt.plot(np.arange(80.1,95.0,0.1), curves[i], label='Experimental',color='red')

        #Plot the pred_curve
        crv = generator.predict([np.array([seqs[i]])])[0]
        for _ in range(1):
            crv = cm.smooth(crv)
        plt.plot(np.linspace(80.1,95.0,len(crv)), crv, label='Pred',color='blue')
        
        #Plot the rev pred_curve
        crv = generator.predict([np.array([rev_seqs[i]])])[0]
        for _ in range(1):
            crv = cm.smooth(crv)
        plt.plot(np.linspace(80.1,95.0,len(crv)), crv, label='R_Pred',color='green')
        
        #Plot the umelt_curve
        plt.plot(np.arange(80.1,95.0,0.1), umelt[i], label='umelt',color='purple')
        
        #Show the plot
        #plt.ylabel('Dissociation')
        plt.legend(loc="upper right")
        #plt.xlabel('Temperature (C)')
        #plt.ylabel('Dissociation') 
        
        if ids[i] == tag:
            tit = '%s_TEST' % ids[i]
        else:
            tit = '%s_TRAIN' % ids[i]
        plt.title(tit)
        #Plot the diff curves
        #Plot the actual curve
        crv = curves[i]
        crv = np.negative(np.diff(crv))        
        ind += 1
        plt.subplot(30,2,ind)
        plt.plot(np.linspace(80.1,95.0,len(crv)), crv, label='Experimental',color='red')
        
        #Plot pred curve
        crv = generator.predict([np.array([seqs[i]])])[0]
        crv = np.negative(np.diff(crv))
        for _ in range(1):
            crv_s = cm.smooth(crv)        
        
        d = cm.calc_distance(crv_s,np.negative(np.diff(curves[i])))
        plt.plot(np.linspace(80.1,95.0,len(crv_s)), crv_s, label='Pred %f' % d,color='blue')
        
        #Plot rev pred curve 
        crv = generator.predict([np.array([rev_seqs[i]])])[0]
        crv = np.negative(np.diff(crv))
        
        for _ in range(1):
            crv_s = cm.smooth(crv)
        
        d = cm.calc_distance(crv_s,np.negative(np.diff(curves[i])))
        plt.plot(np.linspace(80.1,95.0,len(crv_s)), crv_s, label='R_Pred %f' % d,color='green')
        
        #Plot the umelt curve
        crv = umelt[i]
        crv = np.negative(np.diff(crv))
        d = cm.calc_distance(crv,np.negative(np.diff(curves[i])))
        plt.plot(np.linspace(80.1,95.0,len(crv)), crv, label='UMelt %f' % d,color='purple')        
        plt.title(tit)
        plt.legend(loc="upper right")
        
    plt.tight_layout()
    #plt.show()
    plt.savefig(of)
    plt.close()
    #Export
    os.system("aws s3 cp %s s3://dxoracle1/hrm/v3/run_2/%s" % (of,of))

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5, clipnorm=1.0)

def build_generator(curve_len, seq_len):
    seq_input = Input(shape = (seq_len,))
    first = Dense(256)(seq_input)
    first = LeakyReLU(0.2)(first)
    
    second = Dense(512)(first)
    second = LeakyReLU(0.2)(second)
    
    third = Dense(1028)(second)
    third = LeakyReLU(.2)(third)
    
    out = Dense(curve_len)(third)
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
    'S':(1.0 + 0.75)/2,
    'K':(1.0+0.5)/2    
}

sql_command = """
SELECT sequences.org_id, sequences.sequence, curves.norm_curve, umelt_curves.umelt_pred
FROM sequences
INNER JOIN curves ON sequences.id=curves.seq_id
INNER JOIN umelt_curves ON sequences.id=umelt_curves.seq_id;
"""
curve_mat = []
seq_mat = []
rev_seq_mat = []
tag_mat = []
umelt_mat = []

cur.execute(sql_command)
data = cur.fetchall()
for d in data:
    umelt_mat.append(d[3])
    curve_mat.append(d[2])
    seq_mat.append([base_enc[base] for base in d[1]])
    rev_seq_mat.append([base_enc[base] for base in reverse_complement(d[1])])
    tag_mat.append(d[0])
curves = np.array(curve_mat)/100
seqs = np.array(seq_mat)
rev_seqs = np.array(rev_seq_mat)
ids = np.array(tag_mat)
umelt = np.array(umelt_mat)
curve_len = curves.shape[1]
seq_len = seqs.shape[1]

#Split the data for a loocv
kf = KFold(n_splits=len(curves))



#Run each fold
for tg in ['AECA','ENCL', 'ENDU', 'ENGA', 'ESCO']:
    
    train_index= np.where(ids != tg)
    test_index = np.where(ids == tg)

    
    #Split the matrices
    seqs_train, seqs_test = seqs[train_index], seqs[test_index]
    rev_seqs_train, rev_seqs_test = rev_seqs[train_index], rev_seqs[test_index]
    curves_train, curves_test = curves[train_index], curves[test_index]
    ids_train, ids_test = ids[train_index], ids[test_index]
    umelt_train, umelt_test = umelt[train_index], umelt[test_index]
    
    print('Testing %s' % str(ids_test[0]))
    
    # Creating GAN
    generator= build_generator(curve_len, seq_len)
    discriminator= build_discriminator(curve_len, seq_len)
    gan = build_gan(discriminator, generator, seq_len)
    

    for e in range(250000):
        #Predict curves
        gen_images = generator.predict([seqs_train])
        rev_gen_images = generator.predict([rev_seqs_train])

        #Train the discriminator
        #Label real curves
        y_rl = np.random.uniform(low=0.7, high=1.0, size=(len(seqs_train)))
        

        #Turn on the discriminator
        discriminator.trainable = True

        #Train on real curves
        dloss = discriminator.train_on_batch([curves_train, seqs_train], y_rl)
        dloss = discriminator.train_on_batch([curves_train, rev_seqs_train], y_rl)

        # Label fake curves
        y_fk = np.random.uniform(low=0.0, high=0.3, size=(len(seqs_train)))

        #Train on fake curves
        dloss = discriminator.train_on_batch([gen_images,  seqs_train], y_fk)
        dloss = discriminator.train_on_batch([rev_gen_images,  rev_seqs_train], y_fk)
        
        #Train on mislabeled curves
        shuff_seqs = seqs_train.copy()
        np.random.shuffle(shuff_seqs)
        dloss = discriminator.train_on_batch([gen_images,  shuff_seqs], y_fk)
        shuff_seqs = rev_seqs_train.copy()
        np.random.shuffle(shuff_seqs)
        dloss = discriminator.train_on_batch([gen_images,  shuff_seqs], y_fk)

        #Train the generator
        #Turn off the discriminator
        discriminator.trainable = False

        #Train on fake curves
        gloss = gan.train_on_batch([seqs_train], y_rl)
        gloss = gan.train_on_batch([rev_seqs_train], y_rl)        
            
    #plt results
    output_plt(seqs, rev_seqs, umelt, curves, ids, generator, ids_test[0])
    
    #Outut the matrices
    gen_images_test = generator.predict([seqs_test])
    of = '%s_v3_pred_curve.npy' % ids_test[0]
    np.save(of, gen_images_test)
    
    #Export
    os.system("aws s3 cp %s s3://dxoracle1/hrm/v3_run2/%s" % (of,of))