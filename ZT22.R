rm(list=ls())

# functions for indexing tensors
index <- function(A, v) {
  stopifnot(length(v) == length(dim(A)))
  idx <- paste(v, collapse = ", ")
  return(eval(parse(text = paste0("A[", idx, "]"))))
}

`index<-` <- function(A, v, value) {
  stopifnot(length(v) == length(dim(A)))
  stopifnot(is.atomic(value) & length(value) == 1)
  idx <- paste(v, collapse = ", ")
  eval(parse(text = paste0("A[", idx, "] <- value")))
  return(A)
}

edgetypes <- function(E, A, Z, K){
  Et <- array(0, c(K, ncol(E)))
  for(i in 1:ncol(E)){
    e <- E[,i]
    eZ <- unlist(lapply(e,function(x) Z[x]))
    t <- tabulate(eZ)
    t <- c(t, rep(0,K - length(t)))
    Et[, i] = t
  }
  return(Et)
}

edgecounts <- function(E, A, Z, K, t, v){
  Et <- edgetypes(E, A, Z, K)
  inds <- which(sapply(1:ncol(Et), function(i) all(t == Et[,i]) ))
  count = 0
  for (i in inds) {
    e <- E[,i]
    if(v %in% e){
      count <- count + index(A, e) 
    }
  }
  return(count)
}

## Function for log-likelihood related to node v
loglike <- function(E,A,Z,K,t,v,Q,N,pi,M)
{
  output = 0
  for (i in 1:ncol(M)) {
    t <- M[,i]
    tplus <- t
    tplus[Z[v]] <- tplus[Z[v]] + 1
    output <- output + 
      log(dpois(edgecounts(E,A,Z,K,tplus,v), log(N)*index(Q, tplus + 1)*prod(pi^t)/prod(factorial(t))))
  }
  return(output)
}


## data generation
N = 50
d = 3
K = 3

pi = c(1,1,1)
Z <- c(sample(1:K, size = N, replace = TRUE,prob = pi)) ## clustering configuration
Z = Z[order(Z)]

# Link probability tensor
p = 0.2
q = 0.1
Q = array(0,dim = rep(d+1, K))
index(Q, c(3,0,0) + 1) <- p
index(Q, c(0,3,0) + 1) <- p
index(Q, c(0,0,3) + 1) <- p
index(Q, c(1,2,0) + 1) <- q
index(Q, c(1,0,2) + 1) <- q
index(Q, c(2,1,0) + 1) <- q
index(Q, c(0,1,2) + 1) <- q
index(Q, c(2,0,1) + 1) <- q
index(Q, c(0,2,1) + 1) <- q
index(Q, c(1,1,1) + 1) <- q

A = array(0,dim = c(N,N,N)) # adjacency tensor 
# ("upper triangular", coordinates sorted in ascending order)
E = combn(N,d)
for(i in 1:ncol(E)){
    e <- E[,i]
    eZ <- unlist(lapply(e,function(x) Z[x]))
    t <- tabulate(eZ)
    t <- c(t, rep(0,K - length(t)))
    index(A, e) <- rbinom(1,1,prob=index(Q, t + 1))
}

# matrix of valid (sum to d-1) within-hyperedge community counts
library(gtools)
MM <- t(permutations(N, K, 0:N, repeats.allowed = TRUE))
M = MM
offset = 0
for (i in 1:ncol(MM)) {
    t <- MM[,i]
    if(sum(t) != d-1){
      M <- M[,-(i-offset)]
      offset <- offset + 1
    }
}

# Similarity matrix
S = array(0,dim = c(N,N))
for(i in 1:N){
  for(j in i:N){
    count = 0
    for(k in 1:ncol(E)){
      e <- E[,k]
      if (all(c(i,j)%in%e)){
        count <- count + index(A, e)
      }
    }
    S[i,j] <- count
    S[j,i] <- S[i,j]
  }
}  
degs <- diag(S)
for(i in 1:N){
  S[i,i] <- 0
}

eig <- eigen(S)
SK <- eig$vectors[,2:(K+1)]
est <- kmeans(SK, K)$cluster
library(fossil) # contains rand.index
adj.rand.index(est, Z)
Z
est

for (v in 1:N) {
  maxloglike = -Inf
  argmax = 1
  for (k in 1:K) {
    est[v] <- k
    ll <- loglike(E,A,est,K,t,v,Q,N,pi,M)
    if (ll > maxloglike) {
      maxloglike <- ll
      argmax <- k
    }
  }
  est[v] <- argmax
}

adj.rand.index(est, Z)
Z
est

