import numpy as np

data_npy = np.load("test20_dataset.npy", allow_pickle=True)  # allow_pickle=True nếu lưu object
print(type(data_npy))
print(data_npy)
