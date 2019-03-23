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
    parser.add_argument("--batch_size",default=1,type=int)
    parser.add_argument("--title",default="")
    parser.add_argument("--x_label",default="")
    parser.add_argument("--y_label",default="")

    args = parser.parse_args()

    xs = []
    ys = []
    evaluations = np.zeros((1,args.batch_size))
    
    for result in os.listdir(args.directory):

        evaluation = np.load("{}{}/evaluations/{}_MultiDimensional-v0.npy".format(
            args.directory,result,args.policy_name))

        regex = re.search(r'^.*n([0-9.]*)_([0-9]*)$',result)
        print(result)

        if not float(regex.group(1)) in xs:
            xs.append(float(regex.group(1)))
            print(len(xs)-1)
            print(regex.group(2))
            eval_vect = np.zeros((1,args.batch_size))
            eval_vect[0,int(regex.group(2))] = evaluation[-1,0]
            evaluations = np.r_[evaluations,eval_vect]
            print(evaluations)
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            evaluations[idx+1,int(regex.group(2))] = evaluation[-1,0]

        print(evaluation[-1,0])

    evaluations = evaluations[1:,:]
    
    ys = np.mean(evaluations,axis=1)
    errors = np.std(evaluations,axis=1)
    print(ys)

    print(evaluations)

    xs = list(map(float,xs))
    ys = list(map(float,ys))
    data = np.array([xs,ys]).transpose()
    data2 = []
    for row in data:
        data2.append(tuple(row))

    data2 = sorted(data2,key=lambda tup: tup[0])
    data = np.array(data2)
    xs = data[:,0]
    ys = data[:,1]

    plt.errorbar(xs, ys, errors, fmt="--o")
    plt.title(args.title)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.ylim(bottom=0)
    plt.savefig("{}total_scores.png".format(args.directory))
    plt.show()
