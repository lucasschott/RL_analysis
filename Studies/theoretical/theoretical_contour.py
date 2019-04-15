import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def high_reward_expectation(position):
    return 1/(10*(np.abs(1-min(0.9,position))));

def low_reward_expectation(position):
    return 0.1/(10*(np.abs(-1-max(-0.9,position))));


def f(state):
    return np.max([
            high_reward_expectation(state[0]),
            low_reward_expectation(state[0]),
            high_reward_expectation(state[1]),
            low_reward_expectation(state[1])])

if __name__ == "__main__":

    dim_linspaces = [np.linspace(-1,1,20,endpoint=True),np.linspace(-1,1,20,endpoint=True)]
    meshed_dim = np.meshgrid(*dim_linspaces)
    reshaped_meshed_dim = []
    for dim in meshed_dim:
        reshaped_meshed_dim.append(dim.ravel().reshape(-1,1))
    grid = np.hstack(reshaped_meshed_dim)

    Q_values=[]
    for state in grid :
        q_value = [f(state)]
        q_value.extend(state.flatten().tolist())
        Q_values.append(q_value)
    Q_values = np.array(Q_values)

    qs = Q_values[:,0]
    states = Q_values[:,1:]

    fig, ax = plt.subplots(1)
    plt.set_cmap('RdYlGn')

    colorset = ax.tricontourf(states[:,0], states[:,1], qs,levels=199)
    colorbar = fig.colorbar(colorset)
    colorbar.ax.set_ylabel('Q values')
    colorbar.set_clim(0.1,1)

    ax.set_xlabel('1st dimension')
    ax.set_ylabel('2nd dimension')
    ax.set_xticks(np.arange(-1, 1.2, step=0.2))
    ax.set_yticks(np.arange(-1, 1.2, step=0.2))

    plt.show()
    fig.savefig("theoretical_contour.png")
