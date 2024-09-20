import pickle
from collections import deque


class Replay_Buffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)  # Create a deque (double-ended queue) with a maximum length
        self.training_data = []  # List to store the training data

    def add(self, data):
        """
        Add a new data point to the replay buffer.

        Args:
            data: The data point to be added to the buffer.
        """
        self.buffer.append(data)

    def get_training_data(self):
        """
        Get a list of all the data points in the buffer for training.

        Returns:
            list: A list of all the data points in the buffer.
        """
        self.training_data = list(self.buffer)
        return self.training_data

    def clear(self):
        """
        Clear the replay buffer and the training data list.
        """
        self.buffer.clear()
        self.training_data.clear()

    def save_to_disk(self, file_path):
        """
        Save the replay buffer to disk using pickle.

        Args:
            file_path (str): The path to the file where the buffer should be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.buffer, file)

    @staticmethod
    def load_from_disk(file_path):
        """
        Load a replay buffer from disk using pickle.

        Args:
            file_path (str): The path to the file where the buffer is saved.

        Returns:
            Replay_Buffer: A Replay_Buffer instance loaded from the file.
        """
        with open(file_path, 'rb') as file:
            buffer = pickle.load(file)
            replay_buffer = Replay_Buffer(buffer_size=len(buffer))  # Create a new Replay_Buffer instance
            replay_buffer.buffer = buffer  # Assign the loaded buffer to the new instance
            return replay_buffer