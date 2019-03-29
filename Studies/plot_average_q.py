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
    q_values = []
    
    for result in os.listdir(args.directory):

        print(result)
        if re.search(r'^.*visualizations.*$', result):
            continue

        q_value = np.load("{}{}/logs/q_values.npy".format(
            args.directory,result))

        regex = re.search(r'^.*n([0-9.]*)_([0-9]*)$',result)
        
        if not float(regex.group(1)) in xs:
            xs.append(float(regex.group(1)))
            print(len(xs)-1)
            print(regex.group(2))
            eval_vect = np.zeros(args.batch_size).tolist()
            eval_vect[int(regex.group(2))] = q_value
            q_values.append(eval_vect)
            print(q_value.shape)
        else:
            idx = xs.index(float(regex.group(1)))
            print(idx)
            print(regex.group(2))
            q_values[idx][int(regex.group(2))] = q_value
            print(q_value.shape)

    q_values = np.array(q_values)
    q_values = np.mean(q_values,axis=1)
    
    xs = list(map(float,xs))
    ys = list(q_values)
    data = np.array([xs,ys],dtype=object).transpose()
    data2 = []
    for row in data:
        data2.append(tuple(row))

    data2 = sorted(data2,key=lambda tup: tup[0])
    data = np.array(data2,dtype=object)
    xs = data[:,0]
    q_values = data[:,1]

    new = []
    for q in q_values:
        new.append(q)
    q_values = np.array(new)
    print(q_values.shape)
    
    for i,x in enumerate(xs):
        if len(q_values[i])>1:
            vis_2d.visualize_Q_time(q_values[i], save=True,
                    name="Q_contour_time_{}.gif".format(int(x)),
                    title=r'$Q(s,\pi(s))$ ; ' + args.title + ' {}'.format(int(x)),
                    path=args.directory + "/visualizations",
                    steps_name=" ; timestep",
                    steps=np.arange(0, len(q_values[i]))*args.eval_freq,
                    fps=4)
        vis_2d.visualize_Q(q_values[i,-1], save=True,
                name="Q_contour_{}.png".format(int(x)),
                title=r'$Q(s,\pi(s))$ ; ' + args.title + ' {}'.format(int(x)),
                path=args.directory + "/visualizations")
    
    vis_2d.visualize_Q_time(q_values[:,-1], save=True,
            name="Q_contour_time_{}.gif".format(args.title),
            title=r'$Q(s,\pi(s))$',
            path=args.directory + "/visualizations",
            steps_name=args.title,
            steps=xs, fps=2)

