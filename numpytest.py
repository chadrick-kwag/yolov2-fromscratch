import numpy as np 

a=np.array([[1,1],[2,2]])
b = np.array([[2,2],[3,3]])

ret = np.linalg.norm(a-b)

print(ret)

