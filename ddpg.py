import torch

class DDPG:

    def __init__(self):

        # Create main learning models
        self._main_actor = None
        self._main_critic = None

        # Create target learning models
        self._target_actor = None
        self._target_critic = None