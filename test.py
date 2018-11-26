# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 23:35:26 2018

@author: cogillsb
"""
import numpy as np
import pandas as pd
import curve_manip
import settings
from sklearn.metrics import accuracy_score

class Test(object):
    def __init__(self, test_curves, test_seqs, test_tags, train_curves, train_seqs, train_tags):
        '''
        Initiate object with complete reference and test seqs to generate
        curves to test model performance. 
        '''
        self.test_seqs = test_seqs
        self.test_tags = test_tags
        self.test_curves = test_curves
        self.ref_curves = np.concatenate((train_curves, test_curves))
        self.ref_seqs = np.concatenate((train_seqs, test_seqs))
        self.ref_tags = np.concatenate((train_tags, test_tags))
        self.acc_dist_thresh = None
        self.avg_dist_thresh = None
        self.acc_disc_thresh = None
        self.avg_disc_thresh = None
        

    def test_model(self, model_g, model_d, e):
        '''
        Run a couple of test metrics. 
        '''
        settings.write_to_log('Epoch %i' % e)
        df = self.build_df(self.gen_curves(model_g), model_d)        
        df = self.clear_incomp(df)
        self.element_comp_dist(df, e)
        self.element_comp_disc(df, e)

        acc_dist = self.all_el_comp_dist(df, e)
        print(acc_dist)
        acc_disc = self.all_el_comp_disc(df, e)
        print(acc_disc)
        avg_dist, avg_disc = self.get_avg_dist_disc(df)
        print(avg_dist)
        print(avg_disc)
        #self.check_improv(acc_dist, acc_disc, avg_dist, avg_disc)

    def check_improv(self, acc_dist, acc_disc, avg_dist, avg_disc):
        '''
        Stopping algo if no improvement
        '''
        #Set default if need be
        if self.acc_dist_thresh == None:
            self.acc_dist_thresh = acc_dist
            self.avg_dist_thresh = avg_dist
            self.acc_disc_thresh = acc_disc
            self.avg_disc_thresh = avg_disc                        
        
        if (acc_dist < self.acc_dist_thresh
            and avg_dist > self.avg_dist_thresh
            and acc_disc < self.acc_disc_thresh
            and avg_disc > self.avg_disc_thresh):
            #No improvement. Stop program
            settings.write_to_log('No improvement')
            exit()
            
        else:
            self.acc_dist_thresh = acc_dist
            self.avg_dist_thresh = avg_dist
            self.acc_disc_thresh = acc_disc
            self.avg_disc_thresh = avg_disc  
        
    
    def get_avg_dist_disc(self, df):
        '''
        Find the averge distance and discrimination between target
        and prediction
        '''
        #Check the mean target distance
        df_md = df[df['orgs_A'] == df['orgs_B']]
        mn_dist = df_md['dists'].mean()
        settings.write_to_log('mean distance to target: %f' %  mn_dist)
        mn_disc = df_md['disc'].mean()
        settings.write_to_log('mean disc: %f' %  mn_disc)
        
        return mn_dist, mn_disc
            
    def all_el_comp_dist(self, df, e):
        '''
        get an accuracy using all elements for distance
        '''        
        #Get the avg values        
        df_avg = df.groupby(['orgs_A','orgs_B']).dists.mean().reset_index()
        df_avg.to_csv("/output/avg_dist_comp_%i.csv" % e, index = False)
        df_avg = df_avg.loc[df_avg.groupby('orgs_A').dists.idxmin()]
        avg_acc = accuracy_score(df_avg.orgs_A, df_avg.orgs_B)
        settings.write_to_log('dist_met')
        settings.write_to_log('average: %f' %  avg_acc)
        
        return avg_acc
        
    def all_el_comp_disc(self, df, e):
        '''
        get an accuracy using all elements for distance
        '''        
        #Get the avg values        
        df_avg = df.groupby(['orgs_A','orgs_B']).disc.mean().reset_index()
        df_avg.to_csv("/output/avg_disc_comp_%i.csv" % e, index = False)
        df_avg = df_avg.loc[df_avg.groupby('orgs_A').disc.idxmax()]
        avg_acc = accuracy_score(df_avg.orgs_A, df_avg.orgs_B)
        settings.write_to_log('disc_met')
        settings.write_to_log('average: %f' %  avg_acc)
        
        return avg_acc        
            
    def element_comp_dist(self, df, e):
        '''
        get an accurcy by element for the distance metric
        '''
        #Use the distance metric
        df_grp = df.groupby(['orgs_A','orgs_B', 'els']).dists.mean().reset_index()
        df_grp.to_csv("/output/element_dist_comp_%i.csv" % e, index = False)
        settings.write_to_log('dist_met')
        self.output_el_vals(df_grp.groupby(['orgs_A', 'els']).min().reset_index())

        
    def element_comp_disc(self, df, e):
        '''
        get an accurcy by element for the two discrim metric
        '''        
        #Use the discriminator
        df_grp = df.groupby(['orgs_A','orgs_B', 'els']).disc.mean().reset_index()
        df_grp.to_csv("/output/element_disc_comp_%i.csv" % e, index = False)
        settings.write_to_log('disc_met')
        self.output_el_vals(df_grp.groupby(['orgs_A', 'els']).max().reset_index())

    def output_el_vals(self, df_grp):
        '''
        Output the accucracies by element
        '''
        for el in list(set(df_grp['els'].values)):
            df_acc = df_grp[df_grp['els'] == el]
            acc = accuracy_score(list(df_acc['orgs_A'].values), list(df_acc['orgs_B']))
            settings.write_to_log('%s: %f' % (el, acc)) 
            
    def gen_curves(self, model_g):
        '''
        Use the generator to predict curves
        '''
        #Generate predicted images
        #noise = np.random.normal(0, 1, size=[self.test_seqs.shape[0], settings.random_dim])
        gen_imgs = model_g.predict([self.test_seqs])
        
        return curve_manip.smooth_curves(gen_imgs)

    def build_df(self, gen_curves, model_d):
        '''
        Buiilding a data frame that has all the
        pertinent distances and discriminator calls
        '''
        
        dat = {col:[] for col in ['orgs_A', 'orgs_B','els','dists','disc']}        
        for i in range(len(gen_curves)): 
            for j in range(len(self.ref_curves)):
                #Check if elements match
                if self.test_tags[i].split("_")[1] == self.ref_tags[j].split('_')[1]:
                    #list org connection
                    dat['orgs_A'].append(self.test_tags[i].split("_")[0])
                    dat['orgs_B'].append(self.ref_tags[j].split('_')[0])
                    #list element
                    dat['els'].append(self.test_tags[i].split("_")[1])
                    #Find dist
                    dat['dists'].append(curve_manip.calc_distance(gen_curves[i], self.ref_curves[j]))
                    #Find discriminator call 
                    g = np.array([self.test_curves[i]])
                    s = np.array([self.ref_seqs[j]])
                    dat['disc'].append(float(model_d.predict([g, s])))
                    
        return pd.DataFrame(dat)
        
    def clear_incomp(self, df):
        '''
        This removes org comparisons where all the elements are not present.
        '''
        #Get unique elements for comp org in dict
        df_b = df.groupby(['orgs_B']).els.nunique().reset_index()
        df['B_uni'] = df.orgs_B.map(dict(zip(df_b.orgs_B, df_b.els)))
        df_a = df.groupby(['orgs_A']).els.nunique().reset_index()
        df['A_uni'] = df['orgs_A'].map(dict(zip(df_a.orgs_A, df_a.els)))
        df = df[df.A_uni == df.B_uni]
        
        return df.drop(columns = ['A_uni','B_uni'])
    