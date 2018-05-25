import os

def set_cuda_visible_devices(ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ids
