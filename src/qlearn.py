# Module of Q-Learning

import numpy as np
import visual


class QLearning():
    """
    Class of Q-Learning module,
    including Q table and update functions
    """

    def __init__(self, actions, lr=0.1, gamma=0.9, e=0.9):
        """
        Init function of Q-Learning module
        """
        self.actions = actions
        self.learning_rate = lr
        self.reward_decay = gamma
        self.epsilon_greedy = e
        self.q_table = np.zeros(
            (visual.ROWS * visual.COLUMNS, len(self.actions)))

    def choose_action(self, s):
        """
        Choose next action based on state and Q table

        Parameters:
            s - cat state/position

        Returns:
            actions - next action
        """
        s_index = visual.pos2index(s)
        # action selection
        if np.random.uniform() < self.epsilon_greedy:
            q_index = s_index[1] * visual.COLUMNS + s_index[0]
            state_action = self.q_table[q_index, :]
            state_max = state_action == np.max(state_action)
            state_max_index = np.arange(len(self.actions))[state_max]
            action = int(np.random.choice(state_max_index))
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        """
        Q-Learning with environment feedback and update Q table

        Parameters:
            s  - cat state/position
            a  - cat action
            r  - reward
            s_ - cat next state/position
        """
        s_index = visual.pos2index(s)
        q_index = s_index[1] * visual.COLUMNS + s_index[0]
        q_predict = self.q_table[q_index, a]
        if s_ != "terminal":
            s_index_next = visual.pos2index(s_)
            q_index_next = s_index_next[1] * visual.COLUMNS + s_index_next[0]
            q_target = r + self.reward_decay * self.q_table[
                q_index_next, :].max()
        else:
            q_target = r
        self.q_table[q_index, a] += self.learning_rate * (q_target - q_predict)
