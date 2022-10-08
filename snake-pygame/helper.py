import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(data_plots):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Number of Games')
    #plt.ylim(ymin=0)

    for data in data_plots:
        plt.plot(data)
        plt.text(len(data)-1, data[-1], str(data[-1]))
    plt.pause(0.1)