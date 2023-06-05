import os
import logging
import argparse
import numpy as np
import random
import time
from  tail_shape_estimators import pickands_estimator, dedh_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default="0")
parser.add_argument('--samp_size', type=int, default=20000)#per conditional distribution
parser.add_argument('--estimator', type=str, default="pic")#options: pic, dedh
args = parser.parse_args()
sci_samp_nr=format(5*args.samp_size, ".2e")
sci_samp_nr=sci_samp_nr.split('.')[0]+'e'+str(int(sci_samp_nr.split('+')[1]))#NUmber of sumples in scientific notation

try:
    os.mkdir('./logs') 
except OSError as error: 
    print('Folder exists')  
    

try:
    if args.estimator=='pic':
        os.mkdir('./logs/CTE Pic')#The results when using the Pickands estimator will be saves in this directory regardless of the number of samples 
    else:
        os.mkdir('./logs/CTE DEdH')#The results when using the DEdH estimator will be saves in this directory regardless of the number of samples      
except OSError as error: 
    print('Folder exists')  
    
    
try:
    if args.estimator=='pic':
        os.mkdir('./logs/CTE Pic/'+'CTE Pickands ' +sci_samp_nr+ ' samples')#For each chosen number of samples we save the results in s subfolder of the directory of 'CTE Pic'
    else:
        os.mkdir('./logs/CTE DEdH/'+'CTE DEdH ' +sci_samp_nr+ ' samples') #For each chosen number of samples we save the results in s subfolder of the directory of 'CTE DEdH'  
    
except OSError as error: 
    print('Folder exists')  
    

if args.estimator=='pic':
    logging.basicConfig(filename='./logs/CTE Pic/'+'CTE Pickands ' +sci_samp_nr+ ' samples'+'/run'+str(args.run)+'.log', level=logging.INFO)#We save ech run in the speficic folder for the given number of samples/estimator
else:
    logging.basicConfig(filename='./logs/CTE DEdH/'+'CTE DEdH ' +sci_samp_nr+ ' samples'+'/run'+str(args.run)+'.log', level=logging.INFO)#We save ech run in the speficic folder for the given number of samples/estimator
    
random_means=np.random.rand(30)*10-5#create randomly the means of the gaussian distributions whose mixture will determine f(z); there are 30 means 
random_stds=np.random.rand(30)*4#create randomly the standard deviations of the gaussian distributions whose mixture will determine f(z); there are 30 stds

#Create function which receives a number 'n' as input and returns 'n' samples from f(z)
def sample_z(samp_size):
    mean=random.choices(random_means,k=samp_size) #choose unifromly the mean of the gaussian from which to sample
    std=random.choices(random_stds,k=samp_size) #choose unifromly the std of the gaussian from which to sample
    return mean+std*np.random.normal(0, 1, size=samp_size) # combine the mean and the std to define a gaussian and smaple from that gaussian

#Given a z, sample xi(z), where xi(z) is defined below ((((z+2*z**2+2*z**3)*(np.e**(-np.abs(z)))+a)/b+c)/d)
#the argument ksi below is the maximum that xi(z) reaches
def get_ksi_z(ksi, z):
    b=3.8+1.96
    a=-3*b-3.80
    d=1/((7/8)*ksi+29/8)
    c=ksi*d+3
    return (((z+2*z**2+2*z**3)*(np.e**(-np.abs(z)))+a)/b+c)/d
    
#Once we have a given z0 and xi(z0) we can sample 'nrsamp_perz' samples from a distribution whose tail shape is xi(z0)    
def sample_from_Fz(ksi_z, nrsamp_perz):
    arr=[]
    for ksi in ksi_z:
        #print(ksi)
        x=np.random.rand(nrsamp_perz)
        if ksi>0:
            x=x**(-ksi)+1/ksi**4
        elif ksi<0:
            x=x[:100000]#In the case that xi(z)<0 due to limited machine precision we only choose a smaller subset as the largest sampled points are identical (-1/xi(z0)), thus we cannot divide by h2 in DEhD. 
            #However, estimating xi(z_0) is easy in this case as, one can just take the maximal sample 's_max' and xi(z_0)=-1/s_max.
            x=(x**(-ksi)-1)/ksi
            if nrsamp_perz>100000:
                x=np.append(x,np.zeros(nrsamp_perz-100000))
        else:
            x=-np.log(x)
        arr.append(x)
    arr=np.array(arr).reshape(-1,nrsamp_perz).T #since there are 50 z selected in the beginning then we have a matrix of nrsamp_perz x 50000 samples
    return arr

#Gets 50 z samples from the mixture of gaussians f(z); 
#For a given ksi_max gets the corresponding xi(z) values; for each xi(z) it receives 'sample_perz' samples; it uses these samples to estimate the 50 xi(z) values.
#The results for each z are averaged over 10 iterations to get the finale estimate of xi(z) for each z. 
#As per the CTE algorithm the maximum is chosen as the prediction
def pred_ksi_max(ksi_max, iteration_nr, sample_perz, z_nr):
    iteration_hist=[]
    zs=sample_z(z_nr)
    for iteration in range(iteration_nr):
        samples=sample_from_Fz(get_ksi_z(ksi_max, zs),sample_perz)#
        if args.estimator=='pic':
            pred_ksi_max=pickands_estimator(samples,0.99,5)
        elif args.estimator=='dedh':
            pred_ksi_max=dedh_estimator(samples,0.99,5)
        else:
            pred_ksi_max=hills_estimator(samples,0.99,5)
        iteration_hist.append(pred_ksi_max)
        print(iteration)
    iteration_hist=np.array(iteration_hist).reshape(-1,z_nr)
    print('Estimation of '+str(ksi_max)+' is finished: '+str(iteration_hist.mean(0).max()))
    
    return iteration_hist
    
#It runs 'pred_ksi_max(ksi_max, run_nr, sample_perz, z_nr):' for each max_ksi in the grid 'ksi_max_array'. 
ksi_max_array=np.linspace(-4,5,45)
ksi_max_pred_array=[]
for ksi_max in ksi_max_array:
    all_iterations=pred_ksi_max(ksi_max, 10,args.samp_size,50)
    ksi_max_pred_array.append(np.max(all_iterations.mean(0).max()))
    
logging.info(str(ksi_max_pred_array))
