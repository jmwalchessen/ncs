import numpy as np
import matplotlib.pyplot as plt

refimage = np.load("refimage.npy")

plt.imshow(np.log(refimage[:,:,0]))
plt.savefig("refimage.png")