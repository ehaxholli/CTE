import numpy as np
import matplotlib.pyplot as plt
import os

# Get a list of all subdirectories in the current directory
directories = [d for d in os.listdir('.') if os.path.isdir(d)]

# Calculate the number of rows and columns in the grid
num_rows = (len(directories) + 1) // 2
num_cols = 2

# Create a figure to hold all the plots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 6*num_rows))

# Loop through each directory and create a plot for it
for i, directory in enumerate(sorted(directories)):
    # Calculate the row and column index for the subplot
    row_idx = i // num_cols
    col_idx = i % num_cols

    # Change into the directory and get the current working directory
    os.chdir(directory)
    cwd = os.getcwd().split('/')[-1]

    # Load the data from 10 log files
    runs = np.array([])
    for j in range(1, 11):
        with open('run'+str(j)+'.log') as f:
            lines = f.readlines()
            runs = np.append(runs, np.array(lines[0].split(':')[2].split('[')[1].split(']')[0].split(',')).astype(float))
            
    # Reshape the data into 10 rows and multiple columns
    runs = runs.reshape(10, -1)

    # Calculate the mean and standard deviation of the data
    avg_runs = np.nanmean(runs,0)
    std_runs = np.nanstd(runs,0)

    # Create the x-axis values for the plot
    max_ksis = np.linspace(-4, 5, 45)

    # Plot the ground truth line with circle markers
    axs[row_idx, col_idx].plot(max_ksis, max_ksis, c='black', label='Ground Truth')

    # Plot the mean prediction line with a solid line and triangle markers
    axs[row_idx, col_idx].plot(max_ksis, avg_runs, '--', c='orange', label='Prediction Mean')

    # Fill between the standard deviation lines with light orange and label it
    axs[row_idx, col_idx].fill_between(max_ksis, avg_runs-3*std_runs, avg_runs+3*std_runs, alpha=0.1, color='orange', label='3 Standard Deviations')

    # Set the labels and title of the plot
    axs[row_idx, col_idx].set_xlabel('Shape Parameter', fontsize=14)
    axs[row_idx, col_idx].set_ylabel('Shape Parameter', fontsize=14)
    axs[row_idx, col_idx].set_title(cwd, fontsize=16)

    # Set the tick labels for the x-axis and y-axis
    xticks = np.arange(-4, 6, 1)
    yticks = np.arange(-4, 6, 1)
    axs[row_idx, col_idx].set_xticks(xticks)
    axs[row_idx, col_idx].set_yticks(yticks)

    # Set the font size of the tick labels
    axs[row_idx, col_idx].tick_params(axis='both', labelsize=12)

    # Add legend to the plot
    axs[row_idx, col_idx].legend(fontsize=12)

    # Change back to the parent directory
    os.chdir('..')

# Adjust the spacing between subplots and save the figure

plt.subplots_adjust(hspace=0.4)
fig.suptitle('Predicted vs. True Shape Parameter', fontsize=24)
plt.savefig('all_plots.jpg', bbox_inches='tight')
plt.show()

