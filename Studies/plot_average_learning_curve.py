import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np
import gym_hypercube
from gym_hypercube.visualization import vis_2d
import matplotlib.animation as animation


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",default="results/")
    parser.add_argument("--batch_size",default=1,type=int)
    parser.add_argument("--eval_freq",default=1,type=int)
    parser.add_argument("--title",default="")
    
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
            print(evaluation.shape)
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            evaluations[idx][int(regex.group(2))] = evaluation[:,0]
            print(evaluation.shape)

    evaluations = np.array(evaluations)
    mean = np.mean(evaluations,axis=1)
    if args.batch_size==1:
        std = mean
    else:
        std = np.std(evaluations,axis=1)
    
    xs = list(map(float,xs))
    ys = list(mean)
    zs = list(std)
    data = np.array([xs,ys,zs],dtype=object).transpose()
    data2 = []
    for row in data:
        data2.append(tuple(row))

    data2 = sorted(data2,key=lambda tup: tup[0])
    data = np.array(data2,dtype=object)
    xs = data[:,0]
    mean = data[:,1]
    std = data[:,2]
    
    new = []
    for m in mean:
        new.append(m)
    mean = np.array(new)
    new = []
    for s in std:
        new.append(s)
    std = np.array(new)
    
    for i,x in enumerate(xs):
        print(mean[i])
        print(std[i])
        X = np.arange(0, args.eval_freq * len(mean[i]), args.eval_freq)
        plt.figure()
        plt.errorbar(X, mean[i], std[i], fmt="--o")
        plt.title('Reward per step ; '+ args.title +' {}'.format(float(x)))
        plt.xlabel("timesteps")
        plt.ylabel("reward/step")

        plt.savefig(args.directory + "/visualizations/scores_{}.png".format(float(x)))
        
    plt.figure()
    for i,x in enumerate(xs):
        print(mean[i])
        print(std[i])
        X = np.arange(0, len(mean[i]))*args.eval_freq
        plt.errorbar(X, mean[i], std[i], fmt="--o", label="{}".format(float(x)))
    plt.xlabel("timesteps")
    plt.ylabel("reward/step")
    plt.legend()
    plt.savefig(args.directory + "/visualizations/scores_{}.png".format(args.title))

    fig, ax = plt.subplots(1)

    def animate(i):
        fig.clear()
        ax = fig.add_subplot(111)
        ax.errorbar(X, mean[i], std[i], fmt="--o")
        ax.set_title('Reward per step ; '+ args.title +' {}'.format(float(xs[i])))
        ax.set_xlabel("timesteps")
        ax.set_ylabel("reward/step")
    anim = animation.FuncAnimation(fig, animate, interval=200, frames=len(mean))
    anim.save(args.directory + "/visualizations/scores_{}.gif".format(args.title), writer='imagemagick', fps=2)
