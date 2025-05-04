class BaseAgent:
    def agent_init(self, agent_info={}):
        pass

    def agent_start(self, observation):
        pass

    def agent_step(self, reward, observation):
        pass

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass
