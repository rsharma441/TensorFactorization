import pandas as pd
import numpy as np
from numpy import linalg
import random
import math
import time
import argparse

###Initialized Numbers###
N=50
spec = 50
I = 15 #number of pattern matrices
eta = 50
###Helper Functions###

##delete array
def deleteElements(arr):
    nrows = arr.shape[0]
    ncols = arr.shape[1]

    #newarr = arr
    #amount of sparsity
    nremovals = round(.75*arr.size)

    #initialize set to contain unique values in array to remove
    ret = set()

    #make sure elements of array are symmetrically chosen
    while len(ret) < nremovals:
        cord1 = random.randint(0,nrows-1)
        cord2 = random.randint(0,ncols-1)
        if cord1 != cord2:
            ret.add((cord1, cord2))
            ret.add((cord2,cord1))

    #remove chosen elements in array by setting to zero(missing value indicator)
    for i in ret:
        arr[i[0]][i[1]] = 0

    return(arr)


#same as other deletion function without symmmetry constraint for elements
def nonsymm_deleteElements(arr):
    nrows = arr.shape[0]
    ncols = arr.shape[1]

    nremovals = round(.75*arr.size)

    ret = set()
    while len(ret) < nremovals:
        cord1 = random.randint(0,nrows-1)
        cord2 = random.randint(0,ncols-1)

        ret.add((cord1,cord2))

    for i in ret:
        arr[i[0]][i[1]] = 0

    return(arr)

#sgn function
def sgn(n):
    if n > 0:
        return 1
    elif n < 0:
        return -1
    elif n == 0:
        return 0

#projection function
#keeps values in array to predefined interval
def proj(arr):
    arr[arr > 100] = 100
    arr[arr < 1] = 1

    return(arr)




#get relevant alphas(scalars) for tensor
#for a slice of tensor, picks out correct scaling values for pattern matrices
def get_tensoralphas(alpha_list,index):
    tensor_alphas = []
    for i in range(len(alpha_list)):
        al = alpha_list[i][index]
        #print al
        #print 'hi', i
        tensor_alphas.append(float(al))
    return(tensor_alphas)



###Data Creation###
def create_data():

    #parse type of distribution provided in command line
    parser = argparse.ArgumentParser(description='Distribution of generated data.')
    parser.add_argument('distribution',  help='type of distribution')
    args = parser.parse_args()
    typ = args.distribution

    #initialize lists to hold arrays
    l = [] #final data
    creation_patterns = [] #pattern matrices for data generating process

    #create patterns to define tensor slices
    if typ == 'normal':
        for i in range(I):
            p = np.random.normal(70,15,size = (spec,spec))
            p_symm = (p+p.T)/2 #keep matrices symmetric
            creation_patterns.append(p_symm)
    if typ == 'uniform':
        for i in range(I):
            p = np.random.uniform(1,100,size = (spec,spec))
            p_symm = (p+p.T)/2 #keep matrices symmetric
            creation_patterns.append(p_symm)

    #create sparse alphas to linearly combine with creation patterns
    creation_alphas = np.random.uniform(1,100,size = (I,N))
    creation_alphas = nonsymm_deleteElements(creation_alphas)
    creation_alphas = creation_alphas.tolist()
    for i in range(len(creation_alphas)):
        sum_cralphas = sum(creation_alphas[i])
        for j in range(len(creation_alphas[i])):
            creation_alphas[i][j] = creation_alphas[i][j]/sum_cralphas
            #normalize alphas to be between 0 and 1


    #create tensor slices from linear combination of alpha scalars and pattern matrices
    for i in range(N):
        tensoralphas = get_tensoralphas(creation_alphas,i)
        lincombo = creation_patterns[0] * tensoralphas[0]
        for j in range(1,I):
            lincombo += creation_patterns[j] * tensoralphas[j]
        #keep slices in correct range
        lincombo = proj(lincombo)
        l.append(lincombo)
    #print l

    #create data without underlying data generating process
    #for i in range(N):
    #    b=np.random.normal(70,15, size=(spec,spec))
    #    #print b
    #    b_symm=(b+b.T)/2
    #    b_symm = proj(b_symm)
    #    #print b_symm
    #    l.append(b_symm)


    sparse_slices = np.copy(l)

    #make data sparse
    for i in range(len(sparse_slices)):
        sparse_slices[i] = deleteElements(sparse_slices[i])

    return(l, sparse_slices)






#Initialize Variables for optimization
def initialize():
    patternlist = [] #list of coordinates

    #random pattern matrices
    for i in range(I):
        a = np.random.uniform(1,100, size = (spec,spec))
        a_symm = (a+a.T)/2
        patternlist.append(a_symm)

    #linear alpha weights
    alpha = np.random.uniform(0,1,size = (I,N)) #do we need to change range of possible values?


    #list of alpha coordinates to update
    alphalist = alpha.tolist()

    return(patternlist,alphalist)



###Gradients###

#first gradient for pattern matrices
def grad_pi(tensor_slices,pattern_list,alpha_list,pindex):
    slist = []
    for n in range(len(tensor_slices)):
        Tn = tensor_slices[n]
        tensor_alphas = get_tensoralphas(alpha_list,n)
        #print(len(tensor_alphas))
        lincombo = pattern_list[0]*tensor_alphas[0]
        alpha_i = tensor_alphas[pindex]
        for i in range(1,I):
            lincombo+= tensor_alphas[i]*pattern_list[i]
        nonzero = Tn > 0
        nonzero = nonzero*lincombo
        diff = Tn - nonzero
        grad=diff*-alpha_i
        slist.append(grad)
    return(sum(slist))


#print grad_pi(sparse_slices,patternlist,alphalist,0)

#second gradient for pattern matrices
def grad2_pi(tensor_slices,pattern_list,alpha_list,pindex):
    slist = []
    for n in range(len(tensor_slices)):
        tensor_alphas = get_tensoralphas(alpha_list,n)
        alpha_i = tensor_alphas[pindex]
        alphasq = alpha_i**2
        #print 'hi', alphasq
        slist.append(alphasq)
    return sum(slist)

#print grad2_pi(sparse_slices,patternlist,alphalist,0)

#first gradient for alpha scalar
def grad_alpha_in(tensor_slices, pattern_list, alpha_list, tensor_index, alphaindex):


    Tn = tensor_slices[tensor_index]
    alpha = alpha_list[alphaindex][tensor_index]
    #print 'heyo', alpha
    tensor_alphas = get_tensoralphas(alpha_list,tensor_index)
    #print 'hello', tensor_alphas
    #print tensor_alphas
    Pi_index = tensor_alphas.index(alpha)
    Pi = pattern_list[Pi_index]
    #print Pi

    lincombo = pattern_list[0]*tensor_alphas[0]
    for i in range(1,I):
        lincombo += tensor_alphas[i]*pattern_list[i]

    nonzero = Tn > 0
    nonzero = nonzero*lincombo
    #print nonzero
    diff = Tn - nonzero
    #print diff
    diff = diff*-Pi
    #print diff

    s = np.sum(diff)

    return(s)

#second gradient for alpha scalar
def grad2_alpha_in(tensor_slices,pattern_list,alpha_list, tensor_index,alphaindex):

    alpha = alpha_list[alphaindex][tensor_index]
    tensor_alphas = get_tensoralphas(alpha_list,tensor_index)
    Pi_index = tensor_alphas.index(alpha)
    Pi = pattern_list[Pi_index]
    Pisq = Pi**2
    s=np.sum(Pisq)
    #for element in np.nditer(Pisq):
    #    s += element

    return(s)

#print grad2_alpha_in(sparse_slices,patternlist,alphalist,0,0)

#function that returns objective function value
def objective(tensor_slices,pattern_list,alpha_list,lam):
    slist = []
    for n in range(len(tensor_slices)):
        #print tensor_slices[n]
        #print n
        Tn = tensor_slices[n]
        tensor_alphas = get_tensoralphas(alpha_list,n)
        #print(len(tensor_alphas))
        lincombo = pattern_list[0]*tensor_alphas[0]
        for i in range(1,I):
            lincombo+= tensor_alphas[i]*pattern_list[i]
        nonzero = Tn > 0
        nonzero = nonzero*proj(lincombo)
        diff = Tn - nonzero
        #print n
        #print diff
        norm = np.linalg.norm(diff)
        norm = .5*((norm)**2)
        slist.append(norm)
    su_m = sum(slist)
    l1_alpha = lam*np.sum(np.linalg.norm(np.array(alpha_list),ord=1,axis=1))
    return(su_m + l1_alpha)

#print objective(sparse_slices,pattern_list,alpha_list)

#RMSE function as metric for optimization
def rmse(tensor_slices,pattern_list,alpha_list, height):

    slist = []
    for n in range(len(tensor_slices)):
        Tn = tensor_slices[n]
        tensor_alphas = get_tensoralphas(alpha_list,n)


        lincombo = pattern_list[0]*tensor_alphas[0]
        for i in range(1,I):
            lincombo+= tensor_alphas[i]*pattern_list[i]

        #print(len(Tn), len(lincombo))
        diff = Tn - proj(lincombo)
        norm = np.linalg.norm(diff)
        rmse = norm/len(Tn)
        slist.append(rmse)

    return(sum(slist)/height)

#RMSE function for test data
def rmse_test(tensor_slices,pattern_list,alpha_list,height):
    slist = []
    for n in range(len(tensor_slices)):
        Tn = tensor_slices[n]
        tensor_alphas = get_tensoralphas(alpha_list,n)


        lincombo = pattern_list[0]*tensor_alphas[0]
        for i in range(1,I):
            lincombo+= tensor_alphas[i]*pattern_list[i]

        #print(len(Tn), len(lincombo))
        lincombo = proj(lincombo)
        nonzero = Tn > 0
        lincombo = nonzero*lincombo
        diff = Tn - lincombo
        norm = np.linalg.norm(diff)
        rmse = norm/math.sqrt(np.count_nonzero(Tn))
        slist.append(rmse)

    return(sum(slist)/height)

#split tensor slices into training and testing data
def split(tensor_slices):

    elements = np.count_nonzero(tensor_slices)

    train_count = math.ceil(elements*.6) #proportion of split in data
    #print train_count
    nrows = tensor_slices.shape[0]
    ncols = tensor_slices.shape[1]
    ret = set()
    train = np.zeros(shape=tensor_slices.shape)
    test = tensor_slices.copy()


    while len(ret) < train_count:

        #add tuple of array indeces and value from original array
        cord1 = random.randint(0,nrows-1)
        cord2 = random.randint(0,ncols-1)
        if test[cord1][cord2] != 0:
            #print test[cord1][cord2]
            if cord1 != cord2:
                ret.add((cord1,cord2,test[cord1][cord2]))
                ret.add((cord2,cord1,test[cord2][cord1]))
                #print test[cord2][cord1]
            else:
                ret.add((cord1,cord2,test[cord1][cord2]))

        #print test[cord2][cord1]

    #for set of tuples defining training set
    for i in ret:

        #set element of testing array, set to 0(indicating missing value)
        test[i[0]][i[1]] = 0
        #set same index in training array to value of original array element
        train[i[0]][i[1]] = i[2]

    return(train,test)


#optimization function using coordinate gradient descent
def optimization(tensor_slices, pattern_list, alpha_list,lam):

    #print 'Starting alphas are ', alpha_list
    #print 'Starting first pattern matrix is ', pattern_list[0]
    M = 20000
    objective_list = []
    m=0
    while m < M:


        for i in range(len(pattern_list)):

            #set step length for gradient descent
            gamma = 1/(grad2_pi(tensor_slices,pattern_list,alpha_list,i))
            #print gamma

            grad = grad_pi(tensor_slices,pattern_list,alpha_list,i)
            #print 'updated alphas are ', alpha_list
            #print 'pattern gradient is ', grad

            #descend along gradient while projecting into interval
            new_Pi = proj(pattern_list[i] - gamma*grad)
            pattern_list[i] = new_Pi #update new pattern matrix


        for i in range(len(alpha_list)):
            for j in range(len(alpha_list[i])):

                #set step length for gradient descent
                gamma = 1/grad2_alpha_in(tensor_slices,pattern_list,alpha_list,tensor_index=j,alphaindex=i)
                #grad = grad_alpha_in(tensor_slices,pattern_list,alpha_list,tensor_index = j,alphaindex = i)

                #gradient defined by alpha column to indicate tensor slice
                grad = grad_alpha_in(tensor_slices,pattern_list,alpha_list,tensor_index = j, alphaindex = i)
                #print 'gradient alpha is ', grad

                #formula for new alpha scalar
                new_alpha_in = sgn(alpha_list[i][j] - gamma*grad) * max(0,abs(alpha_list[i][j] - gamma*grad)-gamma*lam)
                alpha_list[i][j] = new_alpha_in #update new alpha scalar

        #check objective function value
        objecti = objective(tensor_slices,pattern_list,alpha_list,lam)
        #print 'At ' + str(m) + ' the objective function is ', objecti
        objective_list.append(objecti)

        #if threshold is passed break optimization
        if len(objective_list) > 2:
            if objective_list[m-1] - objective_list[m] < eta:
                break

        m = m + 1


        #time.sleep(2.5)



    #print 'Final alphas are ', alpha_list
    #print 'Final first pattern is ',pattern_list[0]
    #print 'Final rmse is ', rmse(l,pattern_list,alpha_list,height = N)

    #predict tensor slices with identified alphas and pattern matrices
    final_list = []
    for n in range(len(tensor_slices)):
        tensor_alphas = get_tensoralphas(alpha_list,n)


        lincombo = pattern_list[0]*tensor_alphas[0]
        for i in range(1,I):
            lincombo+= tensor_alphas[i]*pattern_list[i]
        final_list.append(proj(lincombo))


    return(pattern_list,alpha_list,final_list,objective_list)

#print optimization(sparse_slices,patternlist,alphalist,10000)
#print 'Originals', l

#optimization with parameter tuning
def super_optimization(tensor_slices):

    #initialize dictionary of lambda sparsity penalty candidates
    prospective_lambdas = {}

    #range of lambdas to search over
    lambdas = range(2500)

    #split data into training and testing tensor slices
    trainList = []
    testList = []
    for i in tensor_slices:
        train, test = split(i)
        trainList.append(train)
        testList.append(test)

    #initialize dictionary of RMSEs associated with optimization with a lambda
    rmseList = {}

    #initialize search threshold
    epsilon = .1
    k = 0

    while True:

        #start search in the middle
        if k == 0:
            current_index = len(lambdas)//2
            upIndex = current_index
            downIndex = current_index
            upLam = lambdas[current_index]
            downLam = lambdas[current_index]

        if k== 1:
            downIndex = current_index//2
            upIndex = current_index + downIndex
            upLam = lambdas[upIndex]
            downLam = lambdas[downIndex]

        else:
            upLam = lambdas[upIndex]
            downLam = lambdas[downIndex]

        #print 'up lambda guess is ', upLam
        #print 'down lambda guess is ', downLam

        ###Parallelize???###
        #optimize with up lambda candidate
        pattern_list,alpha_list = initialize()

        up = optimization(trainList,pattern_list,alpha_list,upLam)
        up_patterns = up[0]
        up_alphas = up[1]

        #optimize with down lambda candidate
        pattern_list,alpha_list = initialize()

        down = optimization(trainList,pattern_list,alpha_list,downLam)
        down_patterns = down[0]
        down_alphas = down[1]

        #check RMSE for prediction relative to test data
        upRMSE = rmse_test(testList,up_patterns,up_alphas,height=N)
        downRMSE = rmse_test(testList,down_patterns,down_alphas,height=N)

        #choose lambda with smallest RMSE and add it to potential final lambdas
        currentRMSE = min(upRMSE,downRMSE)
        if currentRMSE == upRMSE:
            prospective_lambdas[k] = upLam
        else:
            prospective_lambdas[k] = downLam


        #print 'current RMSE = ', currentRMSE

        if len(rmseList) < 2:
            rmseList[k] = currentRMSE

        #check if RMSE from optimization with current lambda is significantnly different from previous best lambdas
        if len(rmseList) >= 2:
            mini = min(rmseList.items(), key = lambda x: x[1])
            rmseList[k] = currentRMSE
            #print 'mini = ', mini

            #if threshold is passed choose lambda with smallest RMSE
            if abs(rmseList[k] - mini[1]) < epsilon:
                if mini[1] < rmseList[k]:
                    realLambda = prospective_lambdas[mini[0]]
                else:
                    realLambda = prospective_lambdas[k]
                break

        #change next lambda to search around based on previous RMSE from up lambda and down lambda
        #if best lambda is up lambda then search in higher range space around up lambda
        if currentRMSE == upRMSE:
            downIndex = upIndex - (upIndex-current_index)//2
            current_index = upIndex
            upIndex = current_index + (current_index - downIndex)

        #if best lambda is down lambda then search in lower range space around down lambda
        else:
            upIndex = downIndex + (current_index-downIndex)//2
            current_index = downIndex
            downIndex = current_index - (upIndex - current_index)

        k = k+1

    final_pattern_list, final_alpha_list = initialize()
    opti = optimization(tensor_slices,final_pattern_list,final_alpha_list,realLambda)
    patterns = opti[0]
    alphas = opti[1]
    final_objective = opti[3]
    print 'Final objective function value is ', final_objective[len(final_objective)-1]
    print 'Sample alphas: ', alphas[0]
    print 'Sample pattern: ', patterns[0]
    #print 'prediction ', opti[2]
    #print 'original ', l
    print 'Final lambda is ', realLambda
    print 'Final rmse is ', rmse(l,patterns,alphas,height=N)
    return(final_objective)



#TESTING#
if __name__ == "__main__":
    #Data Creation
    l, sparse_slices = create_data()

    #Initialization
    patternlist, alphalist = initialize()

    #Optimization
    opti = optimization(sparse_slices,patternlist,alphalist,lam=1000)
    print 'RMSE for this non-tuned run is ', rmse(l,opti[0],opti[1],height=N)
    print 'Final objective function value is ', opti[3][len(opti[3])-1]

    #Full Optimization with Parameter Tuning
    super_opt = super_optimization(sparse_slices)
