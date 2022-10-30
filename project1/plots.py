from math import ceil, sqrt
import matplotlib.pyplot as plt

def boxplot_every_feature(tx):
    """Show a boxplot for every feature in the dataset."""
    plt.figure(figsize=(20, 10))
    plt.boxplot(tx)
    plt.xlabel('Feature')
    plt.ylabel('Deviation from the mean')
    plt.title("Boxplot of each feature")
    plt.show()

def plot_feature_histograms(x, y):
    """Plot the histogram of every feature in one plot.
    The categories are represented by different colors."""
    size = int(ceil(sqrt(x.shape[1])))
    plt.figure(figsize=(20, 15))
    for i in range(x.shape[1]):
        plt.subplot(size, size, i+1)
        plt.hist(x[y == -1, i], bins=50, alpha=0.5, label='-1')
        plt.hist(x[y == 1, i], bins=50, alpha=0.5, label='1')
        plt.legend(loc='upper right')
        plt.title(f'Feature {i}')
    plt.show()