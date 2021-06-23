# How should you set the QP parameters (H, f, A, and b) to solve the soft margin linear SVM classfier problem using an off-the-shelf QP Solver?

As far as I know you can't solve the primal problem of the softmargin with QP solver. The slack variables are getting in the way. You could however do the dual problem, which does not have this problem.

min (1/2) p^T*H*p + f^T*p
 p

st: Ap smaller_equals b

forall i,j: H[i,j] = (y[i]*y[j])* dotprod(x[i], x[j])
f = unit_vector
A = -entity_matrix
b = zero_vector
