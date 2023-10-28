import math
import matplotlib.pyplot as plt
import numpy as np

x_axis = np.array([5, 20, 50, 100])
training = np.array([1.864318079,
1.555660909,
1.294100737,
1.021547014])

validation = np.array([1.901709046,
1.643566269,
1.452402753,
1.268350691,
])


fig, ax = plt.subplots()
ax.plot(x_axis, training, label= "training CE")
ax.plot(x_axis, validation, label= "validation CE")

plt.xlabel("number of hidden units")
plt.ylabel("avg cross entropy after 100 epochs")
plt.legend(loc="upper right")


plt.show()