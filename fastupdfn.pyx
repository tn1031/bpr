import numpy as np
cimport numpy as np
cimport cython

DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t

cpdef c_upd(np.ndarray[DOUBLE_t, ndim=2] U, 
          np.ndarray[DOUBLE_t, ndim=2] V,
          double mult,
          double lr,
          double reg,
          unsigned int u,
          unsigned int i,
          unsigned int j,
          unsigned int factors):
    cdef unsigned int f
    cdef double grad_u, grad
    for f in xrange(factors):
        grad_u = V[i, f] - V[j, f]
        U[u, f] -= lr * (mult * grad_u + reg * U[u, f])
        
        grad = U[u, f]
        V[i, f] -= lr * (mult * grad + reg * V[i, f])
        V[j, f] -= lr * (-mult * grad + reg * V[j, f])
        
    return U, V

