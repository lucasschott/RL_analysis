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
    parser.add_argument("--learning_timesteps",default=None,type=int)
    parser.add_argument("--eval_freq",default=1,type=int)
    parser.add_argument("--title",default="")
    
    parser.set_defaults(log_scale=False)

    args = parser.parse_args()
    
    if not os.path.exists(args.directory + "/visualizations"):
        os.makedirs(args.directory + "/visualizations")

    xs = []
    pi_values = []
    
    for result in os.listdir(args.directory):

        print(result)
        if re.search(r'^.*visualizations.*$', result):
            continue

        pi_value = np.load("{}{}/logs/pi_values.npy".format(
            args.directory,result))

        regex = re.search(r'^.*n([0-9.]*)_([0-9]*)$',result)
        
        if not float(regex.group(1)) in xs:
            xs.append(float(regex.group(1)))
            print(len(xs)-1)
            print(regex.group(2))
            eval_vect = np.zeros(args.batch_size).tolist()
            eval_vect[int(regex.group(2))] = pi_value
            pi_values.append(eval_vect)
            print(pi_value.shape)
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            pi_values[idx][int(regex.group(2))] = pi_value
            print(pi_value.shape)

    pi_values = np.array(pi_values)
    pi_values = np.mean(pi_values,axis=1)
    
    xs = list(map(float,xs))
    ys = list(pi_values)
    data = np.array([xs,ys],dtype=object).transpose()
    data2 = []
    for row in data:
        data2.append(tuple(row))

    data2 = sorted(data2,key=lambda tup: tup[0])
    data = np.array(data2,dtype=object)
    xs = data[:,0]
    pi_values = data[:,1]
    
    new = []
    for pi in pi_values:
        new.append(pi)
    pi_values = np.array(new)
    
    for i,x in enumerate(xs):
        if len(pi_values[i]) > 1:
            vis_2d.visualize_Pi_time(pi_values[i], save=True,
                    name="Pi_arrow_time_{}.gif".format(int(x)),
                    title=r'$\pi(s)$ ; ' + args.title + ' {}'.format(int(x)),
                    path=args.directory + "/visualizations",
                    steps_name=" ; timestep",
                    steps=np.arange(0, len(pi_values[i]))*args.eval_freq,
                    fps=4)
        vis_2d.visualize_Pi(pi_values[i,-1], save=True,
                name="Pi_arrow_{}.png".format(int(x)),
                title=r'$\pi(s)$ ; ' + args.title + ' {}'.format(int(x)),
                path=args.directory + "/visualizations")

    vis_2d.visualize_Pi_time(pi_values[:,-1], save=True,
            name="Pi_arrow_time_{}.gif".format(args.title),
            title=r'$\pi(s)$',
            path=args.directory + "/visualizations",
            steps_name=args.title,
            steps=xs, fps=2)
