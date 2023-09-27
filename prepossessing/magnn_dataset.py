import numpy as np

class MAGNN_Dataset:
    def __init__(self, np_array, batch_size):
        self.np_array = np_array
        self.batch_size = batch_size
        self.last_pos = 0

    def random_shuffle(self):
        self.last_pos = 0
        self.random_shuffle_array = np.random.permutation(self.np_array)

    def get_batch(self):
        if self.last_pos >= self.np_array.shape[0]:
            raise Exception('get all element from dataset')
        result = self.random_shuffle_array[self.last_pos:self.last_pos + self.batch_size]
        self.last_pos += self.batch_size
        return result[:, :-3], result[:, -3:]