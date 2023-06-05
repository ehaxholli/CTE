import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.getcwd().split('/')[-1]
print(cwd)
runs=np.array([])
for i in  range(1,11):
    with open('run'+str(i)+'.log') as f:
        lines = f.readlines()
        runs=np.append(runs,np.array(lines[0].split(':')[2].split('[')[1].split(']')[0].split(',')).astype(float))
runs=runs.reshape(10,-1)
avg_runs=runs.mean(0)
std_runs=runs.std(0)
max_ksis=np.linspace(-4,5,45)

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(max_ksis, max_ksis, c='red', label='Ground Truth')
ax.plot(max_ksis, avg_runs, '-', c='blue', label='Prediction Mean')
ax.plot(max_ksis, avg_runs-3*std_runs, '--', c='blue', alpha=0.8, label='1 Standard Deviation')
ax.plot(max_ksis, avg_runs+3*std_runs, '--', c='blue', alpha=0.8)
ax.set_xlabel('The True Shape Parameter')
ax.set_ylabel('The Predicted Shape Parameter')
ax.set_xlim([-5,6])
ax.set_ylim([-5,6])
plt.legend()
plt.savefig(cwd+'.jpg')
plt.show()
