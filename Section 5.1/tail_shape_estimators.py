import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random


def dedh_estimator(x,thresh,avg_size):
    y=x.copy()
    length=x.shape[0]
    cutoff=int(thresh*length)
    x=-np.sort(-np.sort(x, axis=0)[cutoff:], axis=0)
    h1=np.array([])
    h2=np.array([])
    depth=np.sqrt(length-cutoff)
    for j in np.arange(depth)[-avg_size:]:
        k=int(j+1)
        h1=np.append(h1,(np.log(x)[:k]-np.log(x)[k]).mean(0))
        h2=np.append(h2,((np.log(x)[:k]-np.log(x)[k])**2).mean(0))

        
    h1=h1.reshape(avg_size,-1)
    h1=h1.mean(0)
    
    h2=h2.reshape(avg_size,-1)
    h2=h2.mean(0)
    ksi=1 + h1 + 0.5/(h1**2/h2-1)
    return ksi

def pickands_estimator(x,thresh,avg_size):
    y=x.copy()
    length=x.shape[0]
    cutoff=int(thresh*length)
    x=-np.sort(-np.sort(x, axis=0)[cutoff:], axis=0)
    ksi=np.array([])
    depth=np.sqrt(length-cutoff)
    for j in np.arange(depth)[-avg_size:]:
        k=int(j+1)
        ksi=np.append(ksi,np.log((x[k]-x[2*k])/(x[2*k]-x[4*k]))/np.log(2))
    ksi=ksi.reshape(avg_size,-1)
    ksi=ksi.mean(0)
    return ksi
