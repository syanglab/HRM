# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 14:29:44 2018

@author: cogillsb
"""

##############################################################################
# Module for run settings for testing convenience
##############################################################################

#Map for all file_names
fl_nms = {
        'log':'/output/log.txt',
        'f_crv':'../data/curves_cln.csv',
        'model_gen':'/output/model_gen',
        }
fld_nms = {
        'loc_folder':'../data/all_primer_blast_output/*'
        }
#Network settings
net = {}
epochs = 300000
random_dim = 100

skp_tht = False

#Layers 
lbl_lyrs = [1024, 512, 256, 128]
dec_lyrs = [256, 512, 1024]
enc_lyrs = [1024, 512, 256, 128]

#Params
dropout = 0.3
momentum = 0.8
leaky_relu = 0.2
stddev = 0.02

#General
seed = 42
test_percent = 0.10
chkpt = 5000
batch_size = 50

height_noise = 0.05
width_noise = 5

#Curve stuff
curve_start = 324

#Entrez
entrez_email = 'cogillsb@Stanford.edu'

def write_to_log(text):
    '''
    Simple method for selective logging.
    '''
    fh = open(fl_nms['log'], 'a')
    fh.write("%s\n" % text)
    fh.close()
