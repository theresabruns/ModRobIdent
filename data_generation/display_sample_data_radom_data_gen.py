import numpy as np
import matplotlib.pyplot as plt

data_store_base_path = "./Data/dev_data/"
file_name = "Workspace_simple-2-joint_2023-05-15_22:02:17_samples:_1000.npy"  # data filename
fn = data_store_base_path + file_name

# fn = "./Data/dev_data/Data_2023-05-16_14:31:36_r:10_s:500/Workspace_rob_n:0_j:3_2023-05-16_14:31:36_samples:_500.npy"


data = np.load(fn)
print(data)

position_data = np.delete(data, np.s_[0:3:1], 1)
print(position_data)

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(position_data[:, 0], position_data[:, 1], position_data[:, 2])
plt.show()
