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
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            pi_values[idx][int(regex.group(2))] = pi_value

    pi_values = np.array(pi_values)
    pi_values = np.mean(pi_values,axis=1)
    
    for i,x in enumerate(xs):
        vis_2d.visualize_Pi_time(pi_values[i], save=True, name="Pi_arrow_time_{}.gif".format(int(x)), title=r'$\pi(s)$ : {}'.format(int(x)),path=args.directory + "/visualizations", eval_freq=args.eval_freq)
        vis_2d.visualize_Pi(pi_values[i][-1], save=True, name="Pi_arrow_{}.png".format(int(x)), title=r'$\pi(s)$ : {}'.format(int(x)), path=args.directory + "/visualizations")
