import numpy as np
import matplotlib.pyplot as plt

# create a figure with 9 rows and 5 columns of subplots
fig, axs = plt.subplots(nrows=9, ncols=5, figsize=(15, 27))

# loop over the subplots and generate the plots
for i, ax in enumerate(axs.flatten()):
    ksi = np.linspace(-4, 5, 45)[i]
    b = 3.8 + 1.96
    a = -3 * b - 3.80
    d = 1 / ((7/8) * ksi + 29/8)
    c = ksi * d + 3
    z = np.linspace(-20, 20, 1000)
    y = (((z + 2 * z**2 + 2 * z**3) * (np.e ** (-np.abs(z))) + a) / b + c) / d
    ax.plot(z, y)
    ax.set_title(f'Max Ksi:{ksi:.2f}')
    ax.set_xlabel('z')
    ax.set_ylabel('ksi(z)')
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], 10))

# add some spacing between the subplots
fig.tight_layout(pad=2.0)

# save the figure
plt.savefig('plots.png')
plt.show()
