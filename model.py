class Model:
    
    def __init__(self):
        pass
    def train_step(self, state, action, reward, next_state, done):
        raise RuntimeError('Please override')
    def get_action(self, state):
        raise RuntimeError('Please override')
    def save(self, file_name='model.pth'):
        raise RuntimeError('Please override')
    def load(self, file_name='model.pth'):
        raise RuntimeError('Please override')