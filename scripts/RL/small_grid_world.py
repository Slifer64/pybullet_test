import numpy as np
import enum
import abc

State = (int, int)


# ============================================
# ============= GridWorld class ==============
# ============================================
class GridWorld:

    def __init__(self, grid_size):

        self.m, self.n = grid_size

        self.states = [self.ind2state(ind) for ind in range(self.m * self.n)]

        self.actions = ["north", "east", "south", "west"]

        self.__actions_act = [(-1, 0), (0, 1), (1, 0),
                              (0, -1)]  # translate verbal actions to actual actions in the grid world
        self.__actions_dict = {name: value for (name, value) in
                               zip(self.actions, self.__actions_act)}  # map them in a dictionary

        self.__s_terminal = [(0, 0), (self.m - 1, self.n - 1)]

    def size(self) -> (int, int):

        return self.m, self.n

    def step(self, state: State, action) -> (float, State):

        act = self.__actions_dict.get(action)
        if act == None:
            raise RuntimeError("Invalid action '" + action + "'...")

        if self.isTerminal(state):
            return (0, state)

        i, j = state

        i += act[0]
        j += act[1]

        if i < 0:
            i = 0
        elif i > self.m - 1:
            i = self.m - 1

        if j < 0:
            j = 0
        elif j > self.n - 1:
            j = self.n - 1

        R = -1  # reward
        new_state = (i, j)

        return (R, new_state)

    def isTerminal(self, state: State) -> bool:

        for s_t in self.__s_terminal:
            if state[0] == s_t[0] and state[1] == s_t[1]:
                return True
        else:
            return False

    def state2ind(self, state: State) -> int:

        return state[0] * self.n + state[1]

    def ind2state(self, ind: int) -> State:

        return int(ind / self.n), ind % self.n

    @staticmethod
    def valueFunctionAsMatrix(value_fun, world) -> np.array(float):
        assert isinstance(value_fun, ValueFunction)
        assert isinstance(world, GridWorld)

        vmat = np.zeros(world.size())
        for state in world.states:
            vmat[state[0], state[1]] = value_fun.get(state)
        return vmat

    @staticmethod
    def policyAsMatrix(policy, world) -> np.array(str):
        assert isinstance(policy, Policy)
        assert isinstance(world, GridWorld)

        act_symb = {"north": '↑', "east": '→', "south": '↓', "west": '←'}

        act_mat = np.empty_like(np.zeros(world.size()), dtype=str)
        for state in world.states:
            if world.isTerminal(state):
                act_mat[state[0], state[1]] = 'x'
            else:
                # _, act = policy.getMax(state)
                act_mat[state[0], state[1]] = act_symb.get(policy.getMax(state)[1])
        return act_mat

    def __str__(self):

        return "States: " + str(self.states) + "\n" \
                                               "Actions: " + str(self.actions)


# ================================================
# ============= ValueFunction class ==============
# ================================================
class ValueFunction:

    def __init__(self, states: [State] = None):
        self.state_value_dict = {}
        if states is not None:
            self.state_value_dict = {state: 0.0 for state in states}

    def get(self, state: State) -> float:
        value = self.state_value_dict.get(state)
        if value is None:
            raise RuntimeError("Invalid state!")
        return value

    def set(self, state: State, value: float) -> None:
        if self.state_value_dict.get(state) is None:
            raise RuntimeError("Invalid state!")
        self.state_value_dict[state] = value

    def diff(self, other_value_fun) -> float:

        err = 0.
        for state in self.state_value_dict.keys():
            temp_err = abs(self.get(state) - other_value_fun.get(state))
            if temp_err > err:
                err = temp_err
        return err

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        cp = ValueFunction()
        cp.state_value_dict = self.state_value_dict.copy()
        return cp


# =========================================
# ============= Policy class ==============
# =========================================
class Policy:

    def __init__(self):

        self.__policy = {}
        self.__actions = []

    @classmethod
    def Random(cls, world):

        obj = cls()
        obj.__actions = world.actions.copy()
        prob = 1.0 / len(world.actions)
        obj.__policy = {}
        for state in world.states:
            obj.__policy[state] = {}
            for act in world.actions:
                obj.__policy[state][act] = prob
        return obj

    @classmethod
    def Greedy(cls, world, value_fun: ValueFunction):

        obj = cls()
        obj.__actions = world.actions.copy()
        obj.__policy = {}
        for state in world.states:
            obj.__policy[state] = {}
            best_act = None
            v_best = -1e10
            for act in world.actions:
                _, s_new = world.step(state, act)
                v = value_fun.get(s_new)
                if v > v_best:
                    best_act = act
                    v_best = v
                obj.__policy[state][act] = 0.0

            obj.__policy[state][best_act] = 1.0
        return obj

    def get(self, state, action):

        return self.__policy[state][action]

    def getMax(self, state):

        best_act = None
        best_prob = -1e6
        for action in self.__actions:
            prob = self.get(state, action)
            if prob > best_prob:
                best_prob = prob
                best_act = action

        return best_prob, best_act

    def isEqual(self, other_policy) -> bool:

        for state in self.__policy.keys():
            for act in self.__policy[state].keys():
                if self.__policy[state][act] != other_policy.get(state, act):
                    return False
        else:
            return True

    def copy(self):
        return self.__copy__()

    def __str__(self):

        return str(self.__policy)

    def __copy__(self):
        cp = Policy()
        cp.__policy = self._Policy__policy.copy()
        return cp


# ============= Policy Evaluation ==============
def policyEvaluation(world, policy: Policy, gamma, zero_tol=1e-3, max_iter=100) -> (ValueFunction, float, int):
    assert isinstance(policy, Policy)

    V = ValueFunction(world.states)
    V_new = ValueFunction(world.states)

    # V = np.zeros(world.size())
    # V_new = np.zeros(world.size())

    iter = 0

    n_act = len(world.actions)

    while iter < max_iter:
        for s in world.states:
            v_temp = 0.
            for i, a in enumerate(world.actions):
                R, s_new = world.step(s, a)
                p = policy.get(s, a)
                v_temp += p * (R + gamma * V.get(s_new))
            V_new.set(s, v_temp)
            # V_new[s[0], s[1]] = v_temp

        err = V_new.diff(V)
        V = V_new.copy()

        if err < zero_tol:
            break

        iter += 1

    return V, err, iter


# ============= Policy Iteration ==============
def policyIteration(world, gamma, max_outer_iter=20, max_inner_iter=100, zero_tol=1e-3) -> (Policy, int):

    policy = Policy.Random(world)
    prev_policy = policy.copy()

    iterations = 0

    while iterations < max_outer_iter:
        value_fun, _, _ = policyEvaluation(world, policy, gamma, zero_tol, max_inner_iter)
        policy = Policy.Greedy(world, value_fun)
        if policy.isEqual(prev_policy):
            break
        prev_policy = policy.copy()
        iterations += 1

    return policy, iterations

# =================================
# ============= MAIN ==============
# =================================

if __name__ == '__main__':
    grid_world = GridWorld((4, 4))

    # s = (3,2)
    # act = "north"
    # R, new_s = grid_world.step(s, act)
    # print("state:", s, ", action:", act, ", new_state:", new_s, ", R:", R)

    print("=========== Policy Evaluation ===========")
    V, err, iter = policyEvaluation(grid_world, policy=Policy.Random(grid_world), gamma=1.0,
                                    zero_tol=1e-3, max_iter=150)

    print("Value function:\n", GridWorld.valueFunctionAsMatrix(V, grid_world))
    print("Error:", err, ", iterations:", iter)

    print("=========== Policy Iteration ===========")

    opt_policy, iter = policyIteration(grid_world, gamma=1.0, max_outer_iter=20, max_inner_iter=100, zero_tol=1e-3)
    print("Optimal policy:\n", GridWorld.policyAsMatrix(opt_policy, grid_world))
    print("Iterations:", iter)

    V_opt, err, iter = policyEvaluation(grid_world, policy=opt_policy, gamma=1.0, zero_tol=1e-3, max_iter=150)
    print("Optimal Value function:\n", GridWorld.valueFunctionAsMatrix(V, grid_world))
    print("Error:", err, ", iterations:", iter)

