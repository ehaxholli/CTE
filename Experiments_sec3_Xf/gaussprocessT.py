import numpy as np
import matplotlib.pyplot as plt
import sklearn
from numpy.random import default_rng
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import os
import argparse
from numpy.random import default_rng
#%%
print('initiate_parser')
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='0')
args = parser.parse_args()
run=args.run
#%%

#Takes a matrix nxm; in each column (1-thresh)*100% of the biggest values are kept, and sorted in a decreasing
#order. Then for each column i we calculate \ksi_{k,n}_i. It returns a vector, where entry i is the mean of the
#last 'avg_size' of \ksi_{k,n}_i over k.
def pickands_estimator(x,thresh,avg_size):
    y=x.copy()
    length=x.shape[0]
    cutoff=int(thresh*length)
    x=-np.sort(-np.sort(x, axis=0)[cutoff:], axis=0)
    ksi=np.array([])
    for j in np.arange(np.sqrt(length-cutoff))[-avg_size:]:
        k=int(j+1)
        ksi=np.append(ksi,np.log((x[k]-x[2*k])/(x[2*k]-x[4*k]))/np.log(2))


    ksi=ksi.reshape(avg_size,-1)
    ksi=ksi.mean(0)
    return ksi



#import data
#import a 1 dimensional time series. Keep only the first 'max_size' points. Create data windows through a sliding window of size 'window_size'. 
#Return 'data[indexer]', where each window is a row, while its values make up the row. 
def import_data(max_size, window_size):

    data=np.loadtxt('001_UCR_Anomaly_35000.txt')[:max_size]
    indexer = np.arange(window_size)[None, :] + np.arange(data.shape[0]+1-window_size)[:, None]

    return data[indexer]

#%%


#Variable 'degrees', refers to the grid of possible models. Each entry j in degrees, represents a hyperparameter h_j, which defies a different model j. 
degrees=np.arange(30)+1
#The results will be saved in the array below:
rez=[]
#How many training sets we choose. Each of them produces an individual conditional distribution.
cross_iterations=2000
#We import 10000 windows of data, where each window has size 50. The first 49 values are used for training, in order to predict the 50th value. We refer to this data set, as the entire dataset X.
data=import_data(10000,50)
#We select around 7% of data (this is the set X_f), where half of which will be selected for training randomly in each iteration.
data1=data[::15]
samp_size=data1.shape[0]

#%%
for degree in degrees:


    print('Run: ', run,'Degree: ', degree)

    #Predictions of the model will be saved here. Each row 'i' will correspond to a given training set 'X_T(i)'. The values on that row correspond to predictions on the testing set 'X\X_T(i)'.
    #We refer to these values as the samples of individual distribution i.
    arr=np.array([])
    #The MSE on the training set 'X_T(i)' is saved here, for each training set 'X_T(i)'.
    mse_train=[]


    for i in range(cross_iterations):
        if i%20==0:
            print('i:', i )
        #Permute the rows of the data set data1:
        perm=np.random.permutation(data1.shape[0])
        data2=(data1[perm])

        #Fit X_T(i), where X_T(i) is made up of half the points of X_f chosen randomly
        kernel = 1.0 * RBF(degree)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
        gpr.fit(data2[:samp_size//2,:-1], data2[:samp_size//2,-1])

        #Predict on the entire X and append to arr
        arr=np.append(arr,gpr.predict(data[:,:-1]))
        
        #calculate the predicted values in order to calculate the test MSE on data2.
        mse_train.append(((gpr.predict(data2[samp_size//2:,:-1])-data2[samp_size//2:,-1])**2).mean())


    #Create an array so that so that the samples of individual distribution i, compose row i. This is needed to calculate test variance.
    predictions=arr.copy().reshape(cross_iterations,data.shape[0])
    
    #Calculate variance on the entire X
    var=((predictions.mean(0)-predictions)**2).mean()
    rez.append(var)

    #Calculate MSE on the entire X. This is the MSE to which we refer in the paper.
    mse=((data[:,-1]-predictions)**2).mean()
    rez.append(mse)

    #Calculate Bias on the entire X
    bias=mse-var
    rez.append(bias)


    #Create a copy of arr named arr2
    arr2=np.abs(arr.copy())
    #Fold individual distributions to the positive side. Reshape so that the samples of individual distribution i, compose row i. Transpose so that they compose column i. 
    arr=np.abs(arr.reshape(cross_iterations,data.shape[0])).T
    #Calculate the shape parameter of the tail for each individual distribution
    ksi_var=pickands_estimator(arr,0.95,3)
    #Save the maximal of them
    rez.append(ksi_var.max())

    #Calculate the tail shape parameter of the empirical total loss function distribution. Array arr2 is the union of all samples of individual distributions.
    totksi_var=pickands_estimator(arr2,0.95,3)
    #Save it
    rez.append(totksi_var.max())


    #Calculate the train MSE
    mse_train=np.array(mse_train).mean()
    rez.append(mse_train)


#Reshape the results so that each row j corresponds to model j with hyperparameter h_j. 
rezmat=np.array(rez).reshape(degrees.shape[0],-1)
#save results in a text file:
np.savetxt('GaussResultsT/'+'run_'+run+'.txt',rezmat)
