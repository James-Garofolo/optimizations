import numpy as np

#A = np.array([[1.00001, 1], [1, 1]])
#b1 = np.array([2.00001, 2])
#b2 = np.array([2,2])

#print(np.linalg.cond(A))
#print(np.linalg.solve(A, b1))
#print(np.linalg.solve(A, b2))

a = np.array([[2,2],[2,4]])

print(np.all(np.linalg.eigvals(a) >= 0))