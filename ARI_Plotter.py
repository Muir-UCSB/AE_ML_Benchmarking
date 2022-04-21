import numpy as np
import matplotlib
import pylab as pl
import os
import pandas as pd
import matplotlib.ticker as ticker
from itertools import cycle, islice



if __name__ == "__main__":

    '''
    Set hyperparameters
    '''
    read_path = 'C:/Research/Framework_Benchmarking/accuracy_plots.csv'
    write_path = 'C:/Research/Framework_Benchmarking/Figures'
    os.chdir(write_path)




    '''
    Read-in and Setup
    '''

    data = pd.read_csv(read_path)
    print(data)
    header_list = data.columns
    deltaTheta = data[header_list[0]].tolist()
    cut_off = np.ones(len(deltaTheta))*0.4








    '''
    Make plot
    '''

    colors = np.array(list(islice(cycle(['tab:blue','tab:orange', 'tab:gray','tab:brown',
                                         'tab:red', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']), len(deltaTheta)+1)))
    tick_marks = np.array(list(islice(cycle(['o','>', 's', 'd','v']), len(deltaTheta)+1)))
    lines = np.array(list(islice(cycle(['-','--', '-.', ':']), len(deltaTheta)+1)))


    SMALL_SIZE = 14
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18
    width = 3
    mark_size = 8

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Adjusted Rand index', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.set_xlim(0, max(deltaTheta))
    ax1.set_ylim(-.01, 1)
    ax1.set_xlabel(r'$\Delta\theta$ (Degrees)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)




    for i, header in enumerate(header_list[1:]):
        y = data[header].tolist()
        ax1.plot(deltaTheta, y, color=colors[i], ls=lines[i], linewidth=width,
            label=header, marker=tick_marks[i], markersize=mark_size)



    #ax1.plot(deltaTheta, cut_off, color='black', linewidth=2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: makes plot square
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.2))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.legend(fontsize=SMALL_SIZE)
    pl.show()
