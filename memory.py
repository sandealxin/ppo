import numpy as np
import pickle

class PPOMemory:
    def __init__(self, batch_size=16, memory_limit=1024,default_file_name='savedmemory.pkl'):
        self.states = []
        self.positions = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
        self.memory_limit = memory_limit
        self.default_file_name = default_file_name

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.random.permutation(n_states)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
 
        return np.array(self.states),\
            np.array(self.positions),\
            np.array(self.actions),\
            np.array(self.probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches

    def store_memory(self, state, position, action, prob, val, reward, done):
        print("memory length:", len(self.dones))
        if len(self.dones) >= self.memory_limit:
            print("memory overlength:", len(self.dones))
            # Remove the oldest memory
            self.states.pop(0)
            self.positions.pop(0)
            self.actions.pop(0)
            self.probs.pop(0)
            self.vals.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            
        # Add the new memory
        self.states.append(state)
        self.positions.append(position)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.positions = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        
    def save_to_file(self, filename=None):
        if filename is None:
            filename = self.default_file_name
        elif not filename.endswith(".pkl"):
            filename += ".pkl"

        data = {
            'states': self.states,
            'positions': self.positions,
            'probs': self.probs,
            'vals': self.vals,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones
        }

        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print("done memory ")

    def load_from_file(self, filename=None):
        if filename is None:
            filename = self.default_file_name
        elif not filename.endswith(".pkl"):
            filename += ".pkl"

        try:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
        except EOFError:
            print("EOFError: Ran out of input. The file might be empty or corrupted.")
            return
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return

        self.states = data['states']
        self.positions = data['positions']
        self.probs = data['probs']
        self.vals = data['vals']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        print("done loading memory ")
