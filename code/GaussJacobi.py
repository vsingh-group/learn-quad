# import jax.numpy as jnp
import numpy as jnp
import scipy
from scipy.special import gamma
from scipy.special import jacobi
from scipy.special import legendre, chebyt, hermite, gegenbauer, laguerre
from scipy.special import roots_jacobi

import jax




@jax.jit
def my_jacobi(n, a, b, z):
  assert jnp.issubdtype(n.dtype, jnp.integer)
  z = z.astype(jnp.result_type(float, z.dtype))

  # Wrap scipy function to return the expected dtype.
  _scipy_jacobi = lambda n, a, b, z: scipy.special.eval_jacobi(n, a, b, z).astype(z.dtype)

  # Define the expected shape & dtype of output.
  result_shape_dtype = jax.ShapeDtypeStruct(
      shape=jnp.broadcast_shapes(z.shape),
      dtype=z.dtype)

  return jax.pure_callback(_scipy_jacobi, result_shape_dtype, n, a, b, z, vectorized=True)

# Recursive generation of the Jacobi polynomial of order n
def ori_Jacobi(n,a,b,x):
	x=jnp.array(x)
	return (jacobi(n,a,b)(x))

def Jacobi(n,a,b,x):
	x=jnp.array(x)
	return (jacobi(n,a,b)(x))
	
# Derivative of the Jacobi polynomials
def DJacobi(n,a,b,x,k: int):
	x=jnp.array(x)
	ctemp = gamma(a+b+n+1+k)/(2**k)/gamma(a+b+n+1)
	return (ctemp*Jacobi(n-k,a+k,b+k,x))

	
# Weight coefficients
def GaussJacobiWeights(Q: int,a,b):
	[X , W] = roots_jacobi(Q,a,b)
	return [X, W]
	

# Weight coefficients
def GaussLobattoJacobiWeights(Q: int,a,b):
	W = []
	X = roots_jacobi(Q-2,a+1,b+1)[0]
	if a == 0 and b==0:
		W = 2/( (Q-1)*(Q)*(ori_Jacobi(Q-1,0,0,X)**2) )
		Wl = 2/( (Q-1)*(Q)*(ori_Jacobi(Q-1,0,0,-1)**2) )
		Wr = 2/( (Q-1)*(Q)*(ori_Jacobi(Q-1,0,0,1)**2) )
	else:
		W = 2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,X)**2) )
		Wl = (b+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,-1)**2) )
		Wr = (a+1)*2**(a+b+1)*gamma(a+Q)*gamma(b+Q)/( (Q-1)*gamma(Q)*gamma(a+b+Q+1)*(Jacobi(Q-1,a,b,1)**2) )
	W = jnp.append(W , Wr)
	W = jnp.append(Wl , W)
	X = jnp.append(X , 1)
	X = jnp.append(-1 , X)    
	return [X, W]
	