import os
import logging
import argparse
import numpy as np
import random
import time
from  tail_shape_estimators import pickands_estimator, dedh_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default="0")#Run ID
parser.add_argument('--samp_size', type=int, default=100000)#Number of samples
parser.add_argument('--estimator', type=str, default="pic")#options: pic, dedh
args = parser.parse_args()
sci_samp_nr=format(args.samp_size, ".2e")
sci_samp_nr=sci_samp_nr.split('.')[0]+'e'+str(int(sci_samp_nr.split('+')[1]))#NUmber of sumples in scientific notation

try:
    os.mkdir('./logs') 
except OSError as error: 
    print('Folder exists')  
    

try:
    if args.estimator=='pic':
        os.mkdir('./logs/POT Pic')#The results when using the Pickands estimator will be saves in this directory regardless of the number of samples 
    else:
        os.mkdir('./logs/POT DEdH')#The results when using the DEdH estimator will be saves in this directory regardless of the number of samples      
except OSError as error: 
    print('Folder exists')  
    
    
try:
    if args.estimator=='pic':
        os.mkdir('./logs/POT Pic/'+'Direct POT Pickands ' +sci_samp_nr+ ' samples')#For each chosen number of samples we save the results in s subfolder of the directory of 'POT Pic'
    else:
        os.mkdir('./logs/POT DEdH/'+'Direct POT DEdH ' +sci_samp_nr+ ' samples') #For each chosen number of samples we save the results in s subfolder of the directory of 'POT DEdH'  
    
except OSError as error: 
    print('Folder exists')  
    

if args.estimator=='pic':
    logging.basicConfig(filename='./logs/POT Pic/'+'Direct POT Pickands ' +sci_samp_nr+ ' samples'+'/run'+str(args.run)+'.log', level=logging.INFO)#We save ech run in the speficic folder for the given number of samples/estimator
else:
    logging.basicConfig(filename='./logs/POT DEdH/'+'Direct POT DEdH ' +sci_samp_nr+ ' samples'+'/run'+str(args.run)+'.log', level=logging.INFO)#We save ech run in the speficic folder for the given number of samples/estimator
random_means=np.random.rand(30)*10-5#create randomly the means of the gaussian distributions whose mixture will determine f(z); there are 30 means
random_stds=np.random.rand(30)*4#create randomly the standard deviations of the gaussian distributions whose mixture will determine f(z); there are 30 stds

#Create function which receives a number 'n' as input and returns 'n' samples from f(z)
def sample_z(samp_size):
    mean=random.choices(random_means,k=samp_size)
    std=random.choices(random_stds,k=samp_size)
    return mean+std*np.random.normal(0, 1, size=samp_size)

#Given a z, calculate xi(z), where xi(z) is defined below ((((z+2*z**2+2*z**3)*(np.e**(-np.abs(z)))+a)/b+c)/d)
#the argument ksi_max below is the maximum that xi(z) reaches
def get_ksi_z(ksi_max, z):
    b=3.8+1.96
    a=-3*b-3.80
    d=1/((7/8)*ksi_max+29/8)
    c=ksi_max*d+3
    return (((z+2*z**2+2*z**3)*(np.e**(-np.abs(z)))+a)/b+c)/d
    
#Once we have a given z0 and xi(z0) we can sample 'nrsamp_perz' samples from a distribution whose tail shape is xi(z0)    
def sample_from_Fz(ksi_z, nrsamp_perz):
    ksi_z_pos=ksi_z[ksi_z>0]
    ksi_z_neg=ksi_z[ksi_z<0]
    ksi_z_0=ksi_z[ksi_z==0]
    if ksi_z_0.shape[0]>0:
        x=np.concatenate((np.random.rand(ksi_z_pos.shape[0])**(-ksi_z_pos)+1/ksi_z_pos**4,(np.random.rand(ksi_z_neg.shape[0])**(-ksi_z_neg)-1)/ksi_z_neg,-np.log(ksi_z_0.shape[0])))
    else:
        x=np.concatenate((np.random.rand(ksi_z_pos.shape[0])**(-ksi_z_pos)+1/ksi_z_pos**4,(np.random.rand(ksi_z_neg.shape[0])**(-ksi_z_neg)-1)/ksi_z_neg))
    return x.squeeze()


#Gets 'args.samp_size' number of z samples from the mixture of gaussians f(z); 
#For a given ksi_max, for each iteration: gets the corresponding xi(z) values from z values; for each xi(z) it receives 1 sample; it uses these samples to estimate the shape of the tail of the marginal.
def pred_ksi_max(ksi_max, iteration_nr, sample_perz, z_nr):
    iteration_hist=[]
    zs=sample_z(z_nr)
    for iteration in range(iteration_nr):
        samples=sample_from_Fz(get_ksi_z(ksi_max, zs),sample_perz).squeeze()
        if args.estimator=='pic':
            pred_ksi_max=pickands_estimator(samples,0.995,5)
        elif args.estimator=='dedh':
            pred_ksi_max=dedh_estimator(samples,0.995,5)
        else:
            pred_ksi_max=hills_estimator(samples,0.995,5)
        iteration_hist.append(pred_ksi_max)
        print(iteration, pred_ksi_max)
    iteration_hist=np.array(iteration_hist)
    print('Estimation of '+str(ksi_max)+' is finished: '+str(iteration_hist.mean(0).max()))
    
    return iteration_hist
    

ksi_max_array=np.linspace(-4,5,45)
ksi_max_pred_array=[]
for ksi_max in ksi_max_array:#It runs 'pred_ksi_max(ksi_max, iteration_nr, sample_perz, z_nr):' for each max_ksi in the grid 'ksi_max_array'. 
    all_iterations=pred_ksi_max(ksi_max, 10,1,args.samp_size)
    ksi_max_pred_array.append(all_iterations.mean(0)[0])#The results over 10 iterations are averaged and saved.
    
logging.info(str(ksi_max_pred_array))




