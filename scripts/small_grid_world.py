import numpy as np

def policyIteration(world, gamma, tol_stop, max_iter):

    V = np.zeros((world.m,world.n))
    V_new = np.zeros((world.m,world.n))

    iter = 1

    while True:
        for s in world.States:
            v_temp = 0
            for a,p in zip(world.Actions,world.act_prob):
                R, s_new = world.step(s, a)
                v_temp += p*( R + gamma*V[s_new[0],s_new[1]] )
            V_new[s[0],s[1]] = v_temp

        V_diff = V_new - V
        V = V_new

        if all( v_diff<tol_stop for v_diff in V_diff ):
            break

        iter += 1
        if iter > max_iter:
            break


class GridWorld:

    def __init__(self, grid_size):

        self.m, self.n = grid_size

        self.States = [ ( int( state_id / self.n) , state_id % self.n) for state_id in range(self.m * self.n) ]

        self.act_prob = [0.25, 0.25, 0.25, 0.25]
        self.Actions = [ "north", "east", "south", "west" ]
        self.Actions_act = [ (-1, 0), (0, 1), (1, 0), (0, -1) ]
        self.Actions_dict = { name:value for (name,value) in zip(self.Actions,self.Actions_act) }

        self.s_terminal = [(0,0), (self.m-1, self.n-1)]

    def step(self, state, action):

        act = self.Actions_dict.get(action)
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

        for s_t in self.s_terminal:
            if state[0]==s_t[0] and state[1]==s_t[1]:
                return True
        else:
            return False

    def __str__(self):

        return "States: " + str(self.States) + "\n" \
                "Actions: " + str(self.Actions)


if __name__ == '__main__':

    grid_world = GridWorld((4,4))

    s = (3,2)
    act = "north"
    R, new_s = grid_world.step(s, act)

    print("state:", s, ", action:", act, ", new_state:", new_s, ", R:", R)


    A = np.random.rand(2,2)
    # B = np.random.rand(2,2)
    B = A + 5e-1

    print(A)
    print(B)

    if np.any((A-B)>1e-3):
        print("A != B")
    else:
        print("A == B")

    # print(grid_world)

    # print("Hello world!")