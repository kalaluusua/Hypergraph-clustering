import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
def I(alpha,d,M):
    x=np.linspace(0,1,10000)
    tot = np.zeros_like(x)
    for i in range(d):
        tot += sp.binom(d-1,i)*alpha[i]*(1-np.exp(-(x*(d-1-2*i))))
    m = max(tot)
    idx = np.where(tot == m)
    Ival = m/2**(d-1)*M
    print(Ival,x[idx])
    plt.plot(x,tot)
    return Ival

d = 4
M = 2
#print('Enter the probability co-efficients for the following tensor vectors:')
# The alpha array below is for the tensors [(4,0),(3,1),(2,2),(1,3)]. For a symmetric model, keep the probabilities for (1,3) and (3,1) the same.
#alpha = np.array([20,10,5,10])
a31 = np.linspace(0.001,0.003,11)
a40 = 0.004
a22 = 0.001
Ival = np.zeros_like(a31)
idx = 0
for i in a31:
    alpha = np.array([a40,i,a22,i])*sp.binom(49,3)/np.log(50)
    Ival[idx] = I(alpha,d,M)
    idx = idx+1
print(Ival)