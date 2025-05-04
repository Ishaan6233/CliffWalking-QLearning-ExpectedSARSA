import numpy as np

class Environment:
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.reset()

    def reset(self):
        self.agent_position = self.start_state
        return self.state_to_index(self.agent_position)

    def state_to_index(self, state):
        return state[0] * self.width + state[1]

    def index_to_state(self, index):
        return divmod(index, self.width)

    def step(self, action):
        row, col = self.agent_position
        if action == 0:  # UP
            row = max(row - 1, 0)
        elif action == 1:  # LEFT
            col = max(col - 1, 0)
        elif action == 2:  # DOWN
            row = min(row + 1, self.height - 1)
        elif action == 3:  # RIGHT
            col = min(col + 1, self.width - 1)

        next_position = (row, col)

        if next_position in self.cliff:
            reward = -100
            next_position = self.start_state
        elif next_position == self.goal_state:
            reward = 0
        else:
            reward = -1

        self.agent_position = next_position
        done = next_position == self.goal_state
        return self.state_to_index(self.agent_position), reward, done
