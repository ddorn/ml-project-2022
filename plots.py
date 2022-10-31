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

def plot_feature_histograms(x, y, density: bool=False, bins: int=30):
    """Plot the histogram of every feature in one plot.
    The categories are represented by different colors."""
    size = int(ceil(sqrt(x.shape[1])))
    plt.figure(figsize=(4*size, 4*size))
    higgs = x[y == 1]
    no_higgs = x[y == -1]
    for i in range(x.shape[1]):
        plt.subplot(size, size, i+1)
        plt.hist(no_higgs[:, i], bins=bins, alpha=0.5, label='-1', density=density)
        plt.hist(higgs[:, i], bins=bins, alpha=0.5, label='1', density=density)
        plt.legend(loc='upper right')
        plt.title(f'Feature {i}')
    plt.show()