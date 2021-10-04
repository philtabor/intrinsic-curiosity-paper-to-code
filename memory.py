class Memory:
    def __init__(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.states = []
        self.new_states = []
        self.actions = []

    def remember(self, state, action, new_state, reward, value, log_prob):
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)

    def clear_memory(self):
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.actions = []
        self.new_states = []
        self.states = []

    def sample_memory(self):
        return self.states, self.actions, self.new_states, self.rewards,\
               self.values, self.log_probs
