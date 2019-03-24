import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np
import gym_multi_dimensional
from gym_multi_dimensional.visualization import vis_2d


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",default="results/")
    parser.add_argument("--batch_size",default=1,type=int)
    parser.add_argument("--eval_freq",default=1,type=int)
    
    parser.set_defaults(log_scale=False)

    args = parser.parse_args()
    
    if not os.path.exists(args.directory + "/visualizations"):
        os.makedirs(args.directory + "/visualizations")

    xs = []
    evaluations = []
    
    for result in os.listdir(args.directory):

        print(result)
        if re.search(r'^.*visualizations.*$', result):
            continue

        evaluation = np.load("{}{}/logs/evaluations.npy".format(
            args.directory,result))

        regex = re.search(r'^.*n([0-9.]*)_([0-9]*)$',result)
        
        if not float(regex.group(1)) in xs:
            xs.append(float(regex.group(1)))
            print(len(xs)-1)
            print(regex.group(2))
            eval_vect = np.zeros(args.batch_size).tolist()
            eval_vect[int(regex.group(2))] = evaluation[:,0]
            evaluations.append(eval_vect)
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            evaluations[idx][int(regex.group(2))] = evaluation[:,0]

    evaluations = np.array(evaluations)
    mean = np.mean(evaluations,axis=1)
    std = np.std(evaluations,axis=1)
    
    for i,x in enumerate(xs):
        print(mean[i])
        print(std[i])
        X = np.arange(0, args.eval_freq * len(mean[i]), args.eval_freq)
        plt.figure()
        plt.errorbar(X, mean[i], std[i], fmt="--o")
        plt.title("Average reward per step, learning {}".format(int(x)))
        plt.savefig(args.directory + "/visualizations/scores_{}.png".format(int(x)))
