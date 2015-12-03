import pandas as pd
import numpy as np
import random

N=50

##delete array 
def deleteElements(arr):
	nrows = arr.shape[0]
	ncols = arr.shape[1]

	#newarr = arr
	#amount of sparsity
	nremovals = round(.75*arr.size)

	ret = set()
	while len(ret) < nremovals:
		cord1 = random.randint(0,nrows-1)
		cord2 = random.randint(0,ncols-1)
		if cord1 != cord2:	
			ret.add((cord1, cord2))
			ret.add((cord2,cord1))

	for i in ret:
		arr[i[0]][i[1]] = 0

	return(arr)

#generate symmetric and sparse matrices 
l = []
for i in range(N):
	b=np.random.uniform(1,100, size=(7,7))
	b_symm=(b+b.T)/2
	l.append(b_symm)

sparse_slices = np.copy(l) 
for i in range(len(sparse_slices)):
	sparse_slices[i] = deleteElements(sparse_slices[i])


print(l[0])
print(sparse_slices[0])