#%%
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure
#import os
#
#translation=0
#method='gaussprocess' 
#dataset='X'   
#%%
#itrnr=1000
#allresults=[]
#for filename in os.listdir(os.getcwd()):
#    if filename[:3]=='run':
#        print(filename)
#        print(filename[9:])
#        imported_arr=np.loadtxt(filename).squeeze()
#        ksis=imported_arr[:itrnr]
#        locs=imported_arr[itrnr:]
#        sort_order=np.argsort(ksis)
#        plt.plot(ksis[sort_order])
#        plt.savefig(filename[9:].split('.')[0]+'ksi.jpg')
#        plt.clf()
#        plt.plot(locs[sort_order])
#        plt.savefig(filename[9:].split('.')[0]+'locs.jpg')
#        plt.clf()
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import figure
#import os
#
#translation=0
#method='gaussprocess' 
#dataset='X'
#
#itrnr = 1000
#allresults = []
#plot_count = 0
#fig, axs = plt.subplots(8, 11, figsize=(11, 8))
#for filename in os.listdir(os.getcwd()):
#    if filename[:3] == 'run':
#        print(filename)
#        print(filename[9:])
#        imported_arr = np.loadtxt(filename).squeeze()
#        ksis = imported_arr[:itrnr]
#        locs = imported_arr[itrnr:]
#        sort_order = np.argsort(ksis)
#
#        plot_title = 'Scale Param: '+filename[9:].split('.')[0]
#        axs[plot_count // 11, plot_count % 11].plot(ksis[sort_order])
#        axs[plot_count // 11, plot_count % 11].set_title(plot_title)
#        plot_count += 1
#        axs[plot_count // 11, plot_count % 11].plot(locs[sort_order])
#        axs[plot_count // 11, plot_count % 11].set_title(plot_title)
#        plot_count += 1
#plt.savefig('plots.jpg')
#plt.clf()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from natsort import natsorted

translation=0
method='gaussprocess' 
dataset='X'

itrnr = 1000
allresults = []
plot_count = 0
fig, axs = plt.subplots(11, 8, figsize=(8*5, 11*5))
for filename in natsorted(os.listdir(os.getcwd())):
    if filename[:3] == 'run':
        print(filename)
        print(filename[9:])
        imported_arr = np.loadtxt(filename).squeeze()
        ksis = imported_arr[:itrnr]
        locs = imported_arr[itrnr:]
        sort_order = np.argsort(ksis)
        scale_list=[format(label, '.1e') for label in [int(filename[9:].split('.')[0])]]

        plot_title = 'Scale: '+scale_list[0]+r'. $\xi$ vals'
        axs[plot_count // 8, plot_count % 8].plot(ksis[sort_order])
        axs[plot_count // 8, plot_count % 8].set_title(plot_title, fontsize=14)
        axs[plot_count // 8, plot_count % 8].tick_params(axis='both', which='major', labelsize=14)

        plot_title = 'Scale: '+scale_list[0]+r'. Threshold'
        plot_count += 1
        axs[plot_count // 8, plot_count % 8].plot(locs[sort_order])
        axs[plot_count // 8, plot_count % 8].set_title(plot_title, fontsize=14)
        axs[plot_count // 8, plot_count % 8].tick_params(axis='both', which='major', labelsize=14)

        plot_count += 1
plt.savefig('crosstailvariability.jpg')
plt.clf()


