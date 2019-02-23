import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np
from operator import itemgetter

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",default="results/")
    parser.add_argument("--policy_name",default="DDPG")
    parser.add_argument("--title",default="")
    parser.add_argument("--x_label",default="")
    parser.add_argument("--y_label",default="")

    args = parser.parse_args()

    xs = []
    ys = []
    
    for result in os.listdir(args.directory):

        evaluation = np.load("{}{}/evaluations/{}_MultiDimensional-v0.npy".format(
            args.directory,result,args.policy_name))

        x = re.search(r'^.*n([0-9]*)$',result)
        
        xs.append(x.group(1))
        ys.append(evaluation[-1])

    xs = list(map(int,xs))
    ys = list(map(float,ys))
    data = np.array([xs,ys]).transpose()
    data2 = []
    for row in data:
        data2.append(tuple(row))

    data2 = sorted(data2,key=lambda tup: tup[0])
    data = np.array(data2)
    xs = data[:,0]
    ys = data[:,1]

    plt.plot(xs,ys)
    plt.title(args.title)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.show()
