from ray.rllib.connectors.connector_v2 import ConnectorV2

from ray.rllib.callbacks.callbacks import RLlibCallback

class FilterTeacherCallback(RLlibCallback):
    
    def __init__(self, policies_to_train, get_learning_agent,):
        super().__init__()
        self.policies_to_train=policies_to_train
        self.get_learning_agent=get_learning_agent
    
    def on_sample_end(self, *, env_runner, metrics_logger, samples, **kwargs):
        for ma_ep in samples:
            _, student_mid = self.get_learning_agent(ma_ep, self.policies_to_train)
            for aid in list(ma_ep.agent_episodes.keys()):
                if ma_ep.module_for(aid) != student_mid:
                    ma_ep._del_agent(aid)