#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

translation=3
method='gaussprocess' 
dataset='X'   
#%%
allresults=[]
for filename in os.listdir(os.getcwd()):
    if filename[:3]=='run':
        print(filename)
        print(np.loadtxt(filename).shape)
        allresults.append(np.loadtxt(filename))
rezmat=np.array(allresults)
print(rezmat.shape)
#%%
rezmatmean=np.nanmean(rezmat,0).T[:,1:]
rezmatstd=np.nanstd(rezmat,0).T[:,1:]
print(rezmatmean.shape)
print(rezmatmean)


#%%
##########
#RELATIVE
##########
figure(figsize=(10, 10), dpi=120)
colors=np.array([['red','red','red'],['green','green','green'],['blue','blue','blue']])
print('colors')
print(colors.shape)
legend_names=['variance','MSE','Bias',r'$\xi\ estimated\ via\ CTE$',r'$\xi\ estimated\ via\ POT$','d']
deg_s=0
deg_e=100
j=0

keep=np.array([0,1,0,1,0,0])
for i in range(keep.shape[0]):
    if keep[i]==1:
        a=rezmatmean[i][deg_s:deg_e+1]
        b=a+1*rezmatstd[i][deg_s:deg_e+1]
        c=a-1*rezmatstd[i][deg_s:deg_e+1]
        plt.plot(translation+np.arange(a.shape[0]),((a-a.min())/(a-a.min()).max()), label=legend_names[i], color=colors[j][0])
        plt.plot(translation+np.arange(a.shape[0]),((b-a.min())/(a-a.min()).max()), color=colors[j][1],alpha=0.2,label='One standard deviation')
        plt.plot(translation+np.arange(a.shape[0]),((c-a.min())/(a-a.min()).max()), color=colors[j][1],alpha=0.2)
        plt.fill_between(translation+np.arange(a.shape[0]),((c-a.min())/(a-a.min()).max()),((b-a.min())/(a-a.min()).max()),color=colors[j][2],alpha=0.2)
        xticks_as_strings=(translation+np.arange(a.shape[0])).astype(str)
        plt.xticks(ticks=translation+np.arange(a.shape[0]), labels=xticks_as_strings)
        plt.xticks(rotation=90)
        plt.xlabel('Length Scale parameter', fontsize=18)
        plt.ylabel('Normalized Estimated Values', fontsize=18)
        j=j+1
plt.legend(prop={"size":13.5})
plt.savefig(method+'_relative_'+dataset+'.jpg')
plt.show()
plt.clf()


#%%
##########
#ABSOLUTE
##########
figure(figsize=(10, 10), dpi=120)
colors=np.array([['green','green','green'],['orange','orange','orange'],['blue','blue','blue']])
legend_names=['variance','MSE','Bias',r'$\xi\ estimated\ via\ CTE$',r'$\xi\ estimated\ via\ POT$','d']
deg_s=0
deg_e=100
j=0
keep=np.array([0,0,0,1,1,0])
for i in range(keep.shape[0]):
    if keep[i]==1:
        a=rezmatmean[i][deg_s:deg_e+1]
        b=a+1*rezmatstd[i][deg_s:deg_e+1]
        c=a-1*rezmatstd[i][deg_s:deg_e+1]
        plt.plot(translation+np.arange(a.shape[0]),a, label=legend_names[i], color=colors[j][0])
        plt.plot(translation+np.arange(a.shape[0]),b, color=colors[j][1],alpha=0.2,label='One standard deviation')
        plt.plot(translation+np.arange(a.shape[0]),c, color=colors[j][1],alpha=0.2)
        plt.fill_between(translation+np.arange(a.shape[0]),c,b,color=colors[j][2],alpha=0.2)
        xticks_as_strings=(translation+np.arange(a.shape[0])).astype(str)
        plt.xticks(ticks=translation+np.arange(a.shape[0]), labels=xticks_as_strings)
        plt.xticks(rotation=90)
        plt.xlabel('Length Scale parameter', fontsize=18)
        plt.ylabel('True Estimated Values', fontsize=18)
        j=j+1
plt.plot(translation+np.arange(a.shape[0]),np.ones(rezmatmean[i][deg_s:deg_e+1].shape[0]),color='blue',label=r'$\xi=1$')
plt.legend(prop={"size":13.5})
plt.savefig(method+'_absolute_'+dataset+'.jpg')
plt.show()
plt.clf()
