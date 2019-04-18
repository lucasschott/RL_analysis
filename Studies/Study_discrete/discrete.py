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
            for action in actions:
                next_state_v = next_state(state,action,n_discrete)
                reward_v = reward(next_state_v,n_discrete)
                if reward_v > 0:
                    done = True
                else:
                    done = False
                mdp[state][action]=[(1,next_state_v,reward_v,done)]

    return mdp



if __name__ == "__main__":

    n_discrete=82
    mdp = mdp_hypercube_2(n_discrete)

    """
    ##policy iteration
    values, policy = policy_iteration(mdp,n_discrete**2,4,20,20,0.2)
    """

    ##value iteration
    values, policy = value_iteration(mdp,n_discrete**2,4,100,0.99)

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')
    colorset = ax.imshow(values.reshape((n_discrete,n_discrete)),origin="lower")
    #,norm = matplotlib.colors.LogNorm(), )
    colorbar = fig.colorbar(colorset)
    colorbar.ax.set_ylabel('Q values')
    #colorbar.set_clim(0.1,1)
    plt.xticks([0,5,10,15,20], [-1,-0.5,0,0.5,1])
    plt.yticks([0,5,10,15,20], [-1,-0.5,0,0.5,1])
    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    plt.show()

    fig.savefig("discrete_contour.png")

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
        #plt.plot(x_pos, y_pos, '.', color="black")
        plt.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.03, head_width=0.6)
    plt.xticks([0,20,40,60,80], [-1,-0.5,0,0.5,1])
    plt.yticks([0,20,40,60,80], [-1,-0.5,0,0.5,1])
    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.show()

    fig.savefig("discrete_arrow.png")
