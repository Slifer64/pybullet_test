import numpy as np
import enum


class Policy (enum.Enum):

    MAX = 1
    RANDOM = 2


# ============= Policy Evaluation ==============

def policyEvaluation(world, policy, gamma, zero_tol=1e-3, max_iter=100):

    assert isinstance(policy, Policy)

    V = np.zeros(world.size())
    V_new = np.zeros(world.size())

    iter = 0

    n_act = len(world.Actions)

    while iter < max_iter:
        for s in world.States:
            q = np.zeros(n_act)
            for i,a in enumerate(world.Actions):
                R, s_new = world.step(s, a)
                q[i] = R + gamma*V[s_new[0],s_new[1]]
            if policy == Policy.MAX:
                V_new[s[0], s[1]] = np.amax(q)
            elif policy == Policy.RANDOM:
                V_new[s[0], s[1]] = np.sum(q) / len(q)

        err = np.amax(np.abs(V_new - V))
        V = V_new.copy()

        if err < zero_tol:
            break

        iter += 1

    return V, err, iter


# ============= Policy Iteration ==============
def policyIteration(world, gamma, zero_tol=1e-3, max_iter=100):


# ============================================
# ============= GridWorld class ==============
# ============================================
class GridWorld:

    def __init__(self, grid_size):

        self.m, self.n = grid_size

        self.States = [ ( int( state_id / self.n) , state_id % self.n) for state_id in range(self.m * self.n) ]

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

    V, err, iter = policyEvaluation(grid_world, policy=Policy.RANDOM, gamma=1.0, zero_tol=1e-3, max_iter=150)

    print(V)
    print("Error:",err, ", iterations:",iter)

    # print( grid_world._GridWorld__Actions_act )