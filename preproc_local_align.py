# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:04:24 2018

@author: cogillsb
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 18:37:06 2018

@author: cogillsb
"""

from Bio import Entrez
from Bio.Seq import Seq
from glob import glob
import numpy as np
import settings
import os
import pandas as pd
import curve_manip
from Bio import AlignIO
from Bio.Align.Applications import ClustalwCommandline

Entrez.email = settings.entrez_email

class PreProc(object):
    def __init__(self):
        self.tags = []
        self.seqs = []
        self.curves = []
        self.seq_dic = {}
        self.max_len = 0
        self.encoding = {
        '-':[0,0,0,0],
        'A':[1,0,0,0],
        'C':[0,1,0,0],
        'T':[0,0,1,0],
        'G':[0,0,0,1],
        'W':[0.5,0,0.5,0],
        'Y':[0,0.5,0.5,0],
        'R':[0.5,0,0,0.5],
        }
        self.build_seq_dic()
        self.encode_seqs()
        self.build_dat()
        self.numpy_convert()
        self.norm_curves()

        
    def build_seq_dic(self):
        '''
        Get the sequences from the files 
        with the org tags.
        '''
        loc_fs = glob(settings.fld_nms['loc_folder'])
        for f in loc_fs:
            #Parse out org and el name
            tag = "_".join(os.path.basename(f).split('_')[0:2])
            #Parse out info for seq fetch
            seq_id, coords = self.parse_loc(f)
            self.get_seqs(tag, seq_id, coords)
        #Align sequences and check length
        for k, v in self.seq_dic.items():
            if len(v)>1:
                v=self.align_seqs(v)
                self.max_len = len(v[0]) if len(v[0]) > self.max_len else self.max_len
                self.seq_dic[k] =  v
            
    def parse_loc(self, file_name):
        '''
        Parse out the primer blast file and get
        pertinent info
        '''
        coords = []
        with open(file_name, 'r') as f:        
            for line in f:
                #Get the reference no.
                if line.startswith('>'):
                    seq_id = line.split()[0][1:]
                #Get coordinates    
                if line.startswith('Template'):
                    coords.extend([int(s) for s in line.split() if s.isdigit()])
        #Reshape coords
        coords = [coords[x:x+4] for x in range(0, len(coords) - 3, 4)]

        return seq_id, coords
    
    def get_seqs (self, tag, seq_id, coords):
        '''
        Fetch sequences from database and 
        add them to the seq_dic
        '''
        rtag = tag + "_r"
        if tag not in self.seq_dic.keys():
            self.seq_dic[tag] = []
        
        if rtag not in self.seq_dic.keys():
            self.seq_dic[rtag] = []
            
        for c in coords:
            if (max(c)-min(c))>1500:
                continue
            #Check strand
            strand = 1 if min(c) == c[0] else 2
            #Fetch seq
           
            handle = Entrez.efetch(db='nucleotide',
                         id=seq_id,
                         rettype="fasta",
                         strand=strand,
                         seq_start=min(c), 
                         seq_stop=max(c))
            seq = handle.read()
            #Parsing magic
            seq = ''.join(seq.split('\n')[1:])            
            self.seq_dic[tag].append(seq)
            
            #add rev_comp
            rseq = str(Seq(seq).reverse_complement())
            self.seq_dic[rtag].append(rseq)  

    
    def align_seqs(self, seqs):
        '''
        Align the amplicons        
        '''
        #Write out to a fasta
        with open('../data/align.fasta', 'w') as file:
            i = 0
            for seq in seqs:
                i += 1
                file.write(">%i\n%s\n\n" %(i, seq))
        #Align the seqs
        clustalw_cline = ClustalwCommandline("clustalw2.exe", infile="../data/align.fasta")
        stdout, stderr = clustalw_cline()
        align = AlignIO.read("../data/align.aln", "clustal")
        
        out_seqs = []
        for a in align.format('fasta').split('>')[1:]:
            out_seqs.append(''.join(a.split('\n')[1:]))
        
        return out_seqs
    
    def encode_seqs(self):
        '''
        Pad, embed, and average sequences.
        '''
        for k, v in self.seq_dic.items():
            emb_seqs = []            
            for seq in v:
                #Pad seq
                while len(seq)<self.max_len:
                    seq = seq + "-"
                #Embed
                emb=[]
                for x in seq:
                    emb.extend(self.encoding[x])
            emb_seqs.append(emb)
            if settings.skp_tht:
                emb_seqs = self.skp_tht(emb_seqs)
            #Update the overall data structure
            self.seq_dic[k] = np.average(emb_seqs, axis=0)   
            
    def build_dat(self):
        '''
        Build the datasets for training model
        '''
        df = pd.DataFrame.from_csv(settings.fl_nms['f_crv'], index_col = None)
        for i, r in df.iterrows():
            tag = list(r)[0]
            if tag in self.seq_dic.keys():
                self.tags.append(tag)
                self.seqs.append(self.seq_dic[tag])
                self.curves.append(list(r)[1:])
                #rev
                rtag = tag + "_r"
                self.tags.append(rtag)
                self.seqs.append(self.seq_dic[rtag])
                self.curves.append(list(r)[1:])
    
    def norm_curves(self):
        '''
        Normalize the curves before the run.
        '''
        self.curves = curve_manip.norm_curve(self.curves)
        
    def numpy_convert(self):
        '''
        Convert data structures to numpy arrays
        '''
        self.curves = np.array(self.curves)
        self.seqs = np.array(self.seqs)
        self.tags = np.array(self.tags)