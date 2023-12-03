import math
import matplotlib.pyplot as plt
import numpy as np

x_axis = np.array([16,
                   32,
                   64,
                   128,
                   256,
                   512])

""" train_loss = np.array([5.737,4.406,3.823,3.479,3.242,3.07,2.939,2.816,
                       2.717,2.631,2.555,2.482,2.413,2.346,2.29])
 """
validation_withPT = np.array([0.486,
0.502,
0.514,
0.519,
0.533,
0.546,
])

validation_noPT = np.array([0.476,
0.49,
0.499,
0.507,
0.522,
0.539,
])

 
fig, ax = plt.subplots()
ax.plot(x_axis, validation_withPT, label= "Accuracy with pretraining")
ax.plot(x_axis, validation_noPT, label= "Accuracy without pretraining")

plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title("Validation accracy over training size")


""" fig, ax = plt.subplots()
ax.plot(x_axis, train_loss, label= "training loss")

plt.title("Training loss over 30k iterations", fontdict={'fontname':'Comic Sans MS'})
plt.xlabel("iteration", fontdict={'fontname':'Comic Sans MS'})
plt.ylabel("training loss", fontdict={'fontname':'Comic Sans MS'})
plt.legend(loc="upper right") """



plt.show()