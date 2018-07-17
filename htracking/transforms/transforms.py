import numpy as np
import torch


class ListToNumpy(object):
    # Convet list object into numpy array with fixed shape.

    def __init__(self, max_objects=20):
        self.max_objects = max_objects

    def __call__(self, target):

        target = np.asarray(target)
        num_objects, num_attributes = target.shape
        assert num_objects <= self.max_objects

        filled_target = np.zeros((self.max_objects, num_attributes), np.float32)
        filled_target[:num_objects] = target[:]

        return filled_target


class NumpyToTensor(object):
    # Convet Numpy into Tensor.

    def __init__(self):
        pass

    def __call__(self, target):

        output = torch.from_numpy(target)

        return output


