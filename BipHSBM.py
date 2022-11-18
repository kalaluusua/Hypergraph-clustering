import numpy as np
import scipy.special as sp
from itertools import combinations
import scipy.linalg as sl

# N - Number of left vertices (individuals)
# M - Number of right vertices (activities)
# d - Number of left vertices in each hyperedge
# K - Number of left communities
def dbip_hsbm(N,M,d,K,rng):
    marks,signed_marks = generate_marks_eq(N,rng)
    print(marks)
    num_hypedges = int(sp.binom(N,d))
    print(num_hypedges)
    comms = np.arange(K)
    tensor = np.zeros([num_hypedges,K]) #Tabulates the number of nodes of each community in each hyperedge
    mem_mat = np.zeros([num_hypedges,d],dtype=int)
    e = 0
    idx = 0
    tensor_list = np.zeros([int(sp.binom(d+K-1,d)),K])
    for iter in combinations(np.arange(N),d):
        mem_mat[e,:] = iter
        tensor[e,:] = tensor_type(iter,marks)
        if np.any(np.all(tensor[e,:] == np.array(tensor_list), axis=1)):
            e += 1
        else:
            tensor_list[idx,:]=tensor[e,:]
            idx += 1
            e += 1
    print('Enter the probabilities for the following tensor vectors:')
    idx = 0
    prob = {}
    for t in tensor_list:
        prob[idx] = [t]
        prob[idx].append(float(input(t)))
        idx += 1

    # Generate adjacency matrix
    A = np.zeros([num_hypedges,M])
    randunif = rng.random([num_hypedges,M])
    for e in range(num_hypedges):        
        tt = tensor[e]
        p = get_key(prob,tt)
        A[e,:] = randunif[e,:]<=p
    e_count = np.sum(A,axis=1)
    print(sum(e_count))

    # Generate similarity matrix
    W = np.zeros([N,N],dtype=int)
    for e in range(num_hypedges):
        mems = mem_mat[e]
        for comb in combinations(mems,2):
            W[comb[0],comb[1]] += e_count[e]
            W[comb[1],comb[0]] += e_count[e]
    return(W,marks)

# function to return key for any value
def get_key(dict,val):
    for key, value in dict.items():
        if np.all(val == value[0]):
            return(float(value[1]))
    return "key doesn't exist"

def generate_marks_eq(N,rng):
    idx = rng.choice(N,N//2,replace = False)
    marks = np.zeros(N,dtype=int)
    signed_marks = np.ones(N)*(-1)
    marks[idx] = 1
    signed_marks[idx] = 1
    return(marks,signed_marks)

def tensor_type(e,marks):
    K = max(marks)+1
    tensor = np.zeros(K,dtype=int)
    for i in range(len(e)):
        tensor[marks[e[i]]] += 1
    return(tensor)

def sweep_ev(N,marks,v):
    # Sweep the eigenvectors to find the one with least misclassification
    min_errors = N
    i = 0
    opt_idx = -1
    for x in v.T[:]:
        sigma_clus = x.real>0
        errors = np.logical_xor(sigma_clus,marks)
        if min_errors > min(sum(errors),N-sum(errors)):
            min_errors = min(sum(errors),N-sum(errors))
            opt_idx = i
        i += 1
    return(min_errors,opt_idx)

def A2L(A):
    D=np.diag(sum(A))
    return D-A

def a2normL(A):
    N = len(A)
    I = np.eye(N)
    Dreciprocal = np.diag(1/sum(A))
    rootD = sl.sqrtm(Dreciprocal)
    return I-rootD.dot(A).dot(rootD)

rng = np.random.default_rng()
N = 100
M = 1
d = 3
K = 2
W,marks = dbip_hsbm(N,M,d,K,rng)
print(W)

L = A2L(W)
w,v = np.linalg.eig(W)
w = w.real
idx = w.argsort()[:]
w = w[idx]
v = v[:,idx]
v = v.real

err_ev,idx_spec = sweep_ev(N,marks,v)
print(err_ev,idx_spec,w[idx_spec])

