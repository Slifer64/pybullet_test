import torch
import numpy as np
import torch
from tqdm import tqdm

if __name__ == '__main__':

    # a = 5
    # assert a == 6, "Hello assert!"

    for i in tqdm(range(100000)):
        pass

    np.random.randint()

    # Update Q values - this should be the same update as your greedy agent above
    # YOUR CODE HERE
    i = self.last_action
    self.arm_count[i] += 1
    self.q_values[i] += (reward - self.q_values[i]) * 1.0 / self.arm_count[i]

    # Choose action using epsilon greedy
    # Randomly choose a number between 0 and 1 and see if it's less than self.epsilon
    # (hint: look at np.random.random()). If it is, set current_action to a random action.
    # otherwise choose current_action greedily as you did above.
    # YOUR CODE HERE
    if np.random.random() < self.epsilon:
        current_action = np.random.randint(0, high=len(self.q_values))
    else:
        current_action = argmax(self.q_values)