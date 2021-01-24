import numpy as np
import matplotlib.pyplot as plt

X = np.zeros((1000, 2), dtype='float32')
i = 0
mean_x = 1
mean_y = 1
dir = 1
while i != 1000:
    X[i:(i + 200), 0] = np.random.normal(mean_x, 0.5, 200)
    X[i:(i + 200), 1] = np.random.normal(mean_y, 0.5, 200)

    mean_x = mean_x + 1.2
    mean_y = mean_y + (dir)*1
    dir = -dir
    i = i + 200

plt.scatter(X[:,0], X[:,1])
plt.show()

np.savetxt("2D_gss.csv", X, delimiter=",")
print("Fine")