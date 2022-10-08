import numpy as np
import math

a = 0.001

list = [0.81449294, 0.90249157, 0.90249157, 0.        ]

print(np.array(list) * a)
b = np.exp(np.array(list, dtype = np.float64) / a) / np.sum(np.exp(np.array(list, dtype = np.float64) / a))
print(b)
for i in range(len(b)):
    if math.isnan(b[i]):
        print("NAN")
