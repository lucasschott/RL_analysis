import numpy as np
import matplotlib.pyplot as plt

def high_reward_expectation(position):
    return 1/(10*(np.abs(1-min(0.9,position))));

def low_reward_expectation(position):
    return 0.1/(10*(np.abs(-1-max(-0.9,position))));


if __name__ == "__main__":

    dim_linspaces = [np.linspace(-1,1,10,endpoint=True),np.linspace(-1,1,10,endpoint=True)]
    meshed_dim = np.meshgrid(*dim_linspaces)
    reshaped_meshed_dim = []
    for dim in meshed_dim:
        reshaped_meshed_dim.append(dim.ravel().reshape(-1,1))
    grid = np.hstack(reshaped_meshed_dim)

    Pi_values=[]
    for state in grid :
        pi_value = state.flatten().tolist()
        pi_value.extend([
            1 if high_reward_expectation(state[0]) > low_reward_expectation(state[0]) else -1,
            1 if high_reward_expectation(state[1]) > low_reward_expectation(state[1]) else -1
            ])
        Pi_values.append(pi_value)
    Pi_values = np.array(Pi_values)

    states = Pi_values[:,:2]
    pis = Pi_values[:,2:]

    fig, ax = plt.subplots(1)

    for pi,state in zip(pis,states):

        x_pos , y_pos = state[0],state[1]
        x_pi , y_pi = pi[0]*0.1,pi[1]*0.1

        plt.plot(x_pos, y_pos, '.', color="black")
        plt.arrow(x_pos, y_pos, x_pi, y_pi, color="black", width=0.001, head_width=0.02)

    plt.xlabel('1st dimension')
    plt.ylabel('2nd dimension')
    plt.xticks([-1,0,1], [-1,0,1])
    plt.yticks([-1,0,1], [-1,0,1])

    plt.show()
    fig.savefig("theoretical_arrow.png")

