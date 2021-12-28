import numpy as np
import enum


class Policy:

    def get(self):
        pass


class RandomPolicy(Policy):

    def __init__(self, states, actions):

        n_states = len(states)
        n_act = len(actions)

        # P = np.ones((n_states,n_act)) * (1.0 / n_act)
        # P = { state: np.ones(n_act)/n_act for state in states }

        self.__policy = {}
        for state in states:
            self.__policy[state] = {}
            for act in actions:
                self.__policy[state][act] = 1.0 / n_act

    def get(self, state, action):

        return self.__policy[state][action]

    def __str__(self):

        return str(self.__policy)


# def greedyPolicy(states, actions, V):
#
#     n_states = len(states)
#     n_act = len(actions)
#
#     P = np.zeros((n_states, n_act)) * (1.0 / n_act)
#
#     return P


# ============= Policy Evaluation ==============

def policyEvaluation(world, policy, gamma, zero_tol=1e-3, max_iter=100):

    assert isinstance(policy, Policy)

    V = np.zeros(world.size())
    V_new = np.zeros(world.size())

    iter = 0

    n_act = len(world.Actions)

    while iter < max_iter:
        for s in world.States:
            v_temp = 0.
            for i,a in enumerate(world.Actions):
                R, s_new = world.step(s, a)
                p = policy.get(s,a)
                v_temp += p*(R + gamma*V[s_new[0],s_new[1]])
            V_new[s[0],s[1]] = v_temp

        err = np.amax(np.abs(V_new - V))
        V = V_new.copy()

        if err < zero_tol:
            break

        iter += 1

    return V, err, iter


# ============= Policy Iteration ==============
# def policyIteration(world, gamma, zero_tol=1e-3, max_iter=100):


# ============================================
# ============= GridWorld class ==============
# ============================================
class GridWorld:

    def __init__(self, grid_size):

        self.m, self.n = grid_size

        self.States = [ self.ind2state(ind) for ind in range(self.m * self.n) ]

        self.Actions = [ "north", "east", "south", "west" ]
        
        self.__Actions_act = [ (-1, 0), (0, 1), (1, 0), (0, -1) ] # translate verbal actions to actual actions in the grid world
        self.__Actions_dict = { name:value for (name,value) in zip(self.Actions,self.__Actions_act) } # map them in a dictionary

        self.__s_terminal = [(0,0), (self.m-1, self.n-1)]

    def size(self):

        return self.m, self.n

    def step(self, state, action):

        act = self.__Actions_dict.get(action)
        if act == None:
            raise RuntimeError("Invalid action '" + action + "'...")

        if self.isTerminal(state):
            return (0, state)

        i, j = state

        i += act[0]
        j += act[1]

        if i < 0:
            i = 0
        elif i > self.m-1:
            i = self.m-1

        if j < 0:
            j = 0
        elif j > self.n-1:
            j = self.n-1

        R = -1 # reward
        new_state = (i,j)

        return (R, new_state)

    def isTerminal(self, state):

        for s_t in self.__s_terminal:
            if state[0]==s_t[0] and state[1]==s_t[1]:
                return True
        else:
            return False

    def state2ind(self, state):

        return state[0]*self.n + state[1]

    def ind2state(self, ind):

        return int(ind/self.n) , ind%self.n

    def __str__(self):

        return "States: " + str(self.States) + "\n" \
                "Actions: " + str(self.Actions)


# =================================
# ============= MAIN ==============
# =================================

if __name__ == '__main__':

    grid_world = GridWorld((4,4))

    # s = (3,2)
    # act = "north"
    # R, new_s = grid_world.step(s, act)
    # print("state:", s, ", action:", act, ", new_state:", new_s, ", R:", R)

    # print(RandomPolicy(grid_world.States, grid_world.Actions))

    V, err, iter = policyEvaluation(grid_world, policy=RandomPolicy(grid_world.States, grid_world.Actions), gamma=1.0, zero_tol=1e-3, max_iter=150)

    print(V)
    print("Error:",err, ", iterations:",iter)

    # print( grid_world._GridWorld__Actions_act )