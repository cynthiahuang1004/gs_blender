import numpy as np, matplotlib.pyplot as plt

depth = np.load('renders/session_000/sensor_0000/raw_data/0000.npy')
gt    = np.load('renders/session_000/sensor_0000/raw_data/0000_gt.npy')

fig, axes = plt.subplots(1, 2)
axes[0].imshow(depth, cmap='viridis'); axes[0].set_title('sample depth')
axes[1].imshow(gt,    cmap='viridis'); axes[1].set_title('GT depth')
plt.show()
