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



#Initialize Variables for optimization
pattern_list = [] #list of coordinates
alpha_list = []
I = 10 #number of pattern matrices
for i in range(I):
    a = np.random.uniform(1,100, size = (7,7))
    a_symm = (a+a.T)/2
    pattern_list.append(a_symm)

#linear weights
alpha = np.random.uniform(0,.025,size = (I,N)) #do we need to change range of possible values?
gamma = .5
beta = .5
lamb = 25
omega = range(1,101)

#list of coordinates to update

for i in np.nditer(alpha):
    alpha_list.append(i)

#sgn function
def sgn(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    elif n == 0:
        return 0

#projection function
def proj(arr):
    for element in np.nditer(arr, op_flags=['readwrite']):
        if element < 1:
            element[...] = 1
        elif element > 100:
            element[...] = 100

    return(arr)

#gradients
def grad_pi(tensor_slices, pattern_list, alpha_list, pindex):
	slist = []
	
	for n in range(len(tensor_slices)):
		Tn = tensor_slices[n]
		tensor_alphas = alpha_list[n::N]
		print(len(tensor_alphas))
		lincombo = pattern_list[0]*tensor_alphas[0]
		for i in range(1,I):
			lincombo+= tensor_alphas[i]*pattern_list[i]

		diff = Tn - lincombo
		alpha_i = tensor_alphas[pindex]
		grad = diff*(-1)*alpha_i
		slist.append(grad)

	sumlist = sum(slist)

	return(sumlist)

def grad_alpha():
	slist=[]
	return(slist)


test = grad_pi(sparse_slices,pattern_list,alpha_list,0)
test1 = pattern_list[0] - gamma*test
test2 = proj(test1)
print(test2)

