import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib



def value_iteration(mdp, state_size, action_size, value_iter=10, gamma=0.9):
    V_old = np.zeros((state_size))
    for _ in range(value_iter):
        V_new = np.zeros(state_size)
        for state, _ in enumerate(V_new):
            actions = list(mdp[state].keys())
            action_values = np.zeros(action_size)
            for action in actions:
                temp=0
                for new_state in mdp[state][action]:
                    temp += new_state[0] * (new_state[2] + gamma * V_old[new_state[1]])
                action_values[action]=temp
            V_new[state] = np.max(action_values)
        V_old = V_new
    Policy = np.zeros((state_size, action_size))
    for state, _ in enumerate(V_old):
        argmax = 0
        value = float("-inf")
        actions = list(mdp[state].keys())
        action_values = np.zeros(action_size)
        for action in actions:
            temp=0
            for new_state in mdp[state][action]:
                temp += new_state[0] * (new_state[2] + gamma * V_old[new_state[1]])
            action_values[action]=temp
        if len(mdp[state]) > 0:
            Policy[state][actions[np.argmax(action_values)]] = 1
    return V_old, Policy



def next_state(state,action,n_discrete):
    if action == 0: #left
        if state%n_discrete > 0:
            return state-1
    if action == 1: #right
        if state%n_discrete < n_discrete-1:
            return state+1
    if action == 2: #up
        if state//n_discrete < n_discrete-1:
            return state+n_discrete
    if action == 3: #down
        if state//n_discrete > 0:
            return state-n_discrete
    return state



def reward(state,n_discrete):
    if state%n_discrete == 0 or state//n_discrete == 0:
        return 0.1
    elif state%n_discrete == n_discrete-1 or state//n_discrete == n_discrete-1:
        return 1
    else:
        return 0



def mdp_hypercube_2(n_discrete=21):
    actions = np.arange(4) #0:left 1:right 2:up 3:down
    mdp = dict()
    for lin in range(n_discrete):
        for col in range(n_discrete):
            state = col + lin * n_discrete
            mdp[state]=dict()
            done = reward(state,n_discrete) > 0
            if not done:
                for action in actions:
                    next_state_v = next_state(state,action,n_discrete)
                    reward_v = reward(next_state_v,n_discrete)
                    done = reward_v > 0
                    mdp[state][action]=[(1,next_state_v,reward_v,done)]
    return mdp



def plot_contour(values,n_discrete):
    fig, ax = plt.subplots(1)
    plt.title('$V(s)$',fontsize=12)
    plt.set_cmap('RdYlGn')
    values = values.reshape((n_discrete,n_discrete))
    values = values[1:-1,1:-1]
    colorset = ax.imshow(values,origin="lower")
    colorbar = fig.colorbar(colorset)
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_clim(0.1,1)
    plt.xticks([0,n_discrete//4,n_discrete//2-2,3*n_discrete//4-2,n_discrete-3], [-1,-0.5,0,0.5,1],fontsize=12)
    plt.yticks([0,n_discrete//4,n_discrete//2-2,3*n_discrete//4-2,n_discrete-3], [-1,-0.5,0,0.5,1],fontsize=12)
    ax.set_xlabel('1st dimension',fontsize=12)
    ax.set_ylabel('2nd dimension',fontsize=12)
    plt.show()
    fig.savefig("discrete_contour.png")



def plot_arrow(policy,n_discrete):
    a = np.arange(0,n_discrete,2)
    b = np.arange(0,n_discrete//2,2)
    states = np.arange((n_discrete)**2).reshape((n_discrete,n_discrete))
    states = states[a]
    states = states[b]
    states = states[:,a]
    states = states[:,b]
    states = states[1:-1,1:-1]
    states = states.flatten()
    pis = np.argmax(policy,axis=1).reshape((n_discrete,n_discrete))
    pis = pis[a]
    pis = pis[b]
    pis = pis[:,a]
    pis = pis[:,b]
    pis = pis[1:-1,1:-1]
    pis = pis.flatten()
    fig, ax = plt.subplots(1)
    plt.title('$\Pi(s)$',fontsize=12)
    for pi,state in zip(pis,states):
        x_pos , y_pos = state%(n_discrete) , state//(n_discrete)
        if pi == 0: #left
            x_pi , y_pi = -1 , 0
        if pi == 1: #right
            x_pi , y_pi = 1 , 0
        if pi == 2: #up
            x_pi , y_pi = 0 , 1
        if pi == 3: #down
            x_pi , y_pi = 0 , -1
        plt.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.03, head_width=0.6)
    plt.xticks([0,n_discrete//4,n_discrete//2-2,3*n_discrete//4-2,n_discrete-3], [-1,-0.5,0,0.5,1],fontsize=12)
    plt.yticks([0,n_discrete//4,n_discrete//2-2,3*n_discrete//4-2,n_discrete-3], [-1,-0.5,0,0.5,1],fontsize=12)
    plt.xlabel('1st dimension',fontsize=12)
    plt.ylabel('2nd dimension',fontsize=12)
    plt.show()
    fig.savefig("discrete_arrow.png")



if __name__ == "__main__":
    n_discrete=80
    mdp = mdp_hypercube_2(n_discrete)
    values, policy = value_iteration(mdp,n_discrete**2,4,n_discrete,0.97)
    plot_contour(values,n_discrete)
    plot_arrow(policy,n_discrete)
