
import numpy as np
from agent import BaseAgent

class ExpectedSarsaAgent(BaseAgent):
    def agent_init(self, agent_info):
        self.num_actions = agent_info["num_actions"]
        self.num_states = agent_info["num_states"]
        self.epsilon = agent_info["epsilon"]
        self.step_size = agent_info["step_size"]
        self.discount = agent_info["discount"]
        self.rand_generator = np.random.RandomState(agent_info["seed"])
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, observation):
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, observation):
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        best_action = self.argmax(self.q[state, :])
        policy_probabilities = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
        policy_probabilities[best_action] += (1 - self.epsilon)
        expected_q = np.sum(policy_probabilities * self.q[state, :])

        self.q[self.prev_state, self.prev_action] += self.step_size * (
            reward + self.discount * expected_q - self.q[self.prev_state, self.prev_action])

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        self.q[self.prev_state, self.prev_action] += self.step_size * (
            reward - self.q[self.prev_state, self.prev_action])

    def argmax(self, q_values):
        top = float("-inf")
        ties = []
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)
