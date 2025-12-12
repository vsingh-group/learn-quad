import equinox as eqx
import jax

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from jax.example_libraries import optimizers

import pdb
import math

from jax import random
from jax import config
# config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights, my_jacobi
import time as sys_time
from scipy.special import jn, jn_zeros
from jaxopt import Bisection


import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import pickle
from functools import partial

from model import modulation_net_forward_pass, generic_tanh_forward_pass

# count param for equinox model
def count_parameters_eq(model_name, pytree):
	param_count = sum(x.size for x in jtu.tree_leaves(eqx.filter(pytree, eqx.is_array)))
	print("Number of parameters in {} is: {}".format(model_name, param_count))
	return param_count

# count parameter for pure jax model
def count_parameters_jax_pure(model_name, params):
	param_count = sum(x.size for x in jax.tree_leaves(params))
	print("Number of parameters in {} is: {}".format(model_name, param_count))
	return param_count

def initialize_mlp_xavier(sizes, key):
	keys = random.split(key, len(sizes))

	def initialize_layer(m, n, key):
		w_key, b_key = random.split(key)
		in_dim = m
		out_dim = n
		xavier_stddev = np.sqrt(2/(in_dim + out_dim), dtype=np.float64)
		initializer = jax.nn.initializers.truncated_normal(stddev=xavier_stddev, lower=-2.0, upper=2.0)
		weight = initializer(w_key, (n, m), jnp.float32)
		bias = jnp.zeros((n,1), jnp.float32).reshape(n)
		return weight, bias
		 
	return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def initialize_mlp_scale(sizes, key):
	keys = random.split(key, len(sizes))
	
	def initialize_layer(m, n, key):
		scale = 0.1
		w_key, b_key = random.split(key)
		weight = scale * random.normal(w_key, (n, m))
		bias =  scale * random.normal(b_key, (n,))
		return weight, bias
		 
	return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def initialize_mlp_he(sizes, key):
	keys = random.split(key, len(sizes))

	def initialize_layer(m, n, key):
		w_key, b_key = random.split(key)
		initializer = jax.nn.initializers.he_uniform()
		weight = initializer(w_key, (n, m), jnp.float32)
		bias = initializer(b_key, (n, 1), jnp.float32).reshape(n)
		return weight, bias
		 
	return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def assign_mlp(post_var):
	all_params = []
	w_value, b_value = post_var
	assert len(w_value) == len(b_value)
	for i in range(len(b_value)):
		weight = jnp.asarray(w_value[i], dtype=jnp.float64)
		bias = jnp.asarray(b_value[i])
		param = (weight, bias)
		all_params.append(param)
	return all_params



def get_optimizer(parameter, lr):
	opt_init, opt_update, get_params = optimizers.adam(lr)
	opt_state = opt_init(parameter)
	return opt_state, opt_update, get_params

def Test_fcnx(N_test, x):
	test_total = []
	for n in range(1,N_test+1):
		test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
		test_total.append(test)
	return jnp.asarray(test_total)

def Test_fcny(N_test, y):
	test_total = []
	for n in range(1,N_test+1):
		test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
		test_total.append(test)
	return jnp.asarray(test_total)

def my_Test_fcn(N_test, z):
	test_total = []
	for n in range(1, N_test+1):
		# jac1 = partial(my_jacobi, n+1, 0, 0)
		# e1 = jac1(z)
		# jac2 = partial(my_jacobi, n-1, 0, 0)
		# e2 = jac2(z)
		# test  = e1 - e2
		test  = Jacobi(n+1, 0, 0, z) - Jacobi(n-1, 0, 0, z)
		test_total.append(test)
	return jnp.asarray(test_total)

def my_Test_fcn_x(n,x):
	# jac1 = partial(my_jacobi, n+1, 0, 0)
	# e1 = jac1(x)
	# jac2 = partial(my_jacobi, n-1, 0, 0)
	# e2 = jac2(x)
	# test  = e1 - e2
	test  = Jacobi(n+1, 0, 0, x) - Jacobi(n-1, 0, 0, x)
	return test

def my_Test_fcn_y(n,y):
	# jac1 = partial(my_jacobi, n+1, 0, 0)
	# e1 = jac1(y)
	# jac2 = partial(my_jacobi, n-1, 0, 0)
	# e2 = jac2(y)
	# test  = e1 - e2
	test  = Jacobi(n+1, 0, 0, y) - Jacobi(n-1, 0, 0, y)
	return test

def my_generate_test_fn(XY_quad_train):
	# Compute the test function
	Nelementx = 5
	Nelementy = 5
	xquad  = jnp.array(XY_quad_train)[:,0:1]
	yquad  = jnp.array(XY_quad_train)[:,1:2]
	all_testx = []
	all_testy = []
	for ex in range(Nelementx):
		temp_testx = []
		temp_testy = []
		for ey in range(Nelementy):
			Ntest_elementx = 5
			Ntest_elementy = 5
			testx_quad_element = Test_fcnx(Ntest_elementx, xquad)
			testy_quad_element = Test_fcny(Ntest_elementy, yquad)
			temp_testx.append(testx_quad_element)
			temp_testy.append(testy_quad_element)
		all_testx.append(temp_testx)
		all_testy.append(temp_testy)
	return all_testx, all_testy

def dTest_fcn(N_test, x):
	d1test_total = []
	d2test_total = []
	for n in range(1,N_test+1):
		if n==1:
			d1test = ((n+2)/2)*Jacobi(n,1,1,x)
			d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
			d1test_total.append(d1test)
			d2test_total.append(d2test)
		elif n==2:
			d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
			d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
			d1test_total.append(d1test)
			d2test_total.append(d2test)    
		else:
			d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
			d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
			d1test_total.append(d1test)
			d2test_total.append(d2test)    
	return jnp.asarray(d1test_total), jnp.asarray(d2test_total)



def plot_all(x_test_plot, y_test_plot, u_test_plot, u_pred_plot, exp_name, epoch):
	fig, axs = plt.subplots(1, 3, figsize=(15, 5))
	c1 = axs[0].contourf(x_test_plot, y_test_plot, u_test_plot, 100, cmap='jet', origin='lower')
	fig.colorbar(c1, ax=axs[0])
	c2 = axs[1].contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
	fig.colorbar(c2, ax=axs[1])
	c3 = axs[2].contourf(x_test_plot, y_test_plot, abs(u_test_plot-u_pred_plot), 100, cmap='jet', origin='lower')
	fig.colorbar(c3, ax=axs[2])
	plt.tight_layout()
	plt.savefig(''.join([exp_name,'/','combine_',str(epoch),'.png']))
	plt.close()


def plot(x_test_plot, y_test_plot, u_pred_plot, exp_name, epoch):
	fontsize = 32
	labelsize = 26
	fig_pred, ax_pred = plt.subplots(constrained_layout=True)
	CS_pred = ax_pred.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
	cbar = fig_pred.colorbar(CS_pred, shrink=0.67)
	cbar.ax.tick_params(labelsize = labelsize)
	ax_pred.locator_params(nbins=8)
	ax_pred.set_xlabel('$x$' , fontsize = fontsize)
	ax_pred.set_ylabel('$y$' , fontsize = fontsize)
	plt.tick_params( labelsize = labelsize)
	ax_pred.set_aspect(1)
	fig_pred.set_size_inches(w=11,h=11)
	plt.savefig(''.join(['./te3e/',exp_name,'/','jaxe_P2D_Predict_',str(epoch),'.png']))
	plt.close()

def plot3d(x, y, z1, z2, epoch, exp_name):
	# Create a figure with two 3D subplots in one row
	fig = plt.figure(figsize=(12, 6))

	# First 3D subplot
	ax1 = fig.add_subplot(121, projection='3d')
	ax1.scatter(x, y, z1, c='blue', marker='o')
	ax1.set_xlabel('X Axis')
	ax1.set_ylabel('Y Axis')
	ax1.set_zlabel('Z1 Axis')
	ax1.set_title('True')

	# Second 3D subplot
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.scatter(x, y, z2, c='green', marker='s')
	ax2.set_xlabel('X Axis')
	ax2.set_ylabel('Y Axis')
	ax2.set_zlabel('Z2 Axis')
	ax2.set_title('Predicted')

	file_name = './te3e/'+exp_name+'/3d/point_',str(epoch),'.pickle'
	pickle.dump(fig, open(file_name, 'wb'))
	return

def plot3d_interactive(x, y, z1, z2, epoch, exp_name):
	# Create a subplot with 1 row and 2 columns
	fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['True', 'Predicted'],
						specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

	# Add first 3D subplot
	fig.add_trace(go.Scatter3d(x=x, y=y, z=z1, mode='markers', marker=dict(color='blue', size=5)), row=1, col=1)

	# Add second 3D subplot
	fig.add_trace(go.Scatter3d(x=x, y=y, z=z2, mode='markers', marker=dict(color='green', size=5)), row=1, col=2)

	# Update layout
	fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'),
					height=400, width=800, title_text='Combined Subplots')

	# Save the plot as an HTML file
	plt.savefig(''.join(['./te3e/',exp_name,'/','jaxe_P2D_Predict_',str(epoch),'.png']))
	file_name = './te3e/'+exp_name+'/3d/point_',str(epoch),'.html'
	fig.write_html(file_name)
	return 


# assumed exact solution, to find f_ext in \Laplacian(u) = f_ext
def u_ext(x, y):
	omegax = 2*jnp.pi
	omegay = 2*jnp.pi
	r1 = 10
	utemp = (0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * jnp.sin(omegay*(y))
	return utemp

# Laplacian(u_ext) = f_ext
def f_ext(x,y):
	omegax = 2*jnp.pi
	omegay = 2*jnp.pi
	r1 = 10
	gtemp = (-0.1*(omegax**2)*jnp.sin(omegax*x) - (2*r1**2)*(jnp.tanh(r1*x))/((jnp.cosh(r1*x))**2))*jnp.sin(omegay*(y))\
			+(0.1*jnp.sin(omegax*x) + jnp.tanh(r1*x)) * (-omegay**2 * jnp.sin(omegay*(y)) )
	return gtemp

def generate_boundary_pt(N_bound):
	# Boundary points
	x_up = 2*lhs(1,N_bound)-1
	y_up = np.empty(len(x_up))[:,None]
	y_up.fill(1)
	b_up = np.empty(len(x_up))[:,None]
	b_up = u_ext(x_up, y_up)                 # 
	x_up_train = np.hstack((x_up, y_up))     # (80,2) , boundary point
	u_up_train = b_up                        # (80,1) , boundary value

	x_lo = 2*lhs(1,N_bound)-1
	y_lo = np.empty(len(x_lo))[:,None]
	y_lo.fill(-1)
	b_lo = np.empty(len(x_lo))[:,None]
	b_lo = u_ext(x_lo, y_lo)
	x_lo_train = np.hstack((x_lo, y_lo))
	u_lo_train = b_lo

	y_ri = 2*lhs(1,N_bound)-1
	x_ri = np.empty(len(y_ri))[:,None]
	x_ri.fill(1)
	b_ri = np.empty(len(y_ri))[:,None]
	b_ri = u_ext(x_ri, y_ri)
	x_ri_train = np.hstack((x_ri, y_ri))
	u_ri_train = b_ri    

	y_le = 2*lhs(1,N_bound)-1
	x_le = np.empty(len(y_le))[:,None]
	x_le.fill(-1)
	b_le = np.empty(len(y_le))[:,None]
	b_le = u_ext(x_le, y_le)
	x_le_train = np.hstack((x_le, y_le))
	u_le_train = b_le    
	
	X_u_train = jnp.concatenate((x_up_train, x_lo_train, x_ri_train, x_le_train))     # jax array, (4*N_bound, 2), 2 for x and y
	u_train = jnp.concatenate((u_up_train, u_lo_train, u_ri_train, u_le_train))       # jax array of boundary value
	print(f'{X_u_train.shape=}')
	print(f'{u_train.shape=}')
	return X_u_train, u_train

def generate_residual_pt(N_residual):
	# Residual points for PINNs
	grid_pt = lhs(2, N_residual)
	xf = 2*grid_pt[:,0]-1
	yf = 2*grid_pt[:,1]-1
	ff = np.asarray([ f_ext(xf[j],yf[j]) for j in range(len(yf))])
	X_f_train = jnp.array(np.hstack((xf[:,None],yf[:,None])))
	f_train = jnp.array(ff[:,None])
	print(f'{X_f_train.shape=}')
	print(f'{f_train.shape=}')
	return X_f_train, f_train

def generate_quadrature(N_quad):
	# Quadrature points
	[X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
	# WX_quad = jnp.ones_like(WX_quad)*(1/N_quad)
	Y_quad, WY_quad   = (X_quad, WX_quad)
	xx, yy            = jnp.meshgrid(X_quad,  Y_quad)                               # tensor product of node and weight
	wxx, wyy          = jnp.meshgrid(WX_quad, WY_quad)
	XY_quad_train     = jnp.hstack((xx.flatten()[:,None],  yy.flatten()[:,None]))   # then arrange them as node and weight, (100, 2)
	WXY_quad_train    = jnp.hstack((wxx.flatten()[:,None], wyy.flatten()[:,None]))                                        # (100, 2)
	print(f'{XY_quad_train.shape=}')
	print(f'{WXY_quad_train.shape=}')
	return XY_quad_train, WXY_quad_train


def generate_1D_subdomain(NE_x):
	# Construction of RHS for VPINNs
	[x_l, x_r] = [-1, 1]
	delta_x    = (x_r - x_l)/NE_x
	grid_x     = jnp.asarray([ x_l + i*delta_x for i in range(NE_x+1)])         # array([-1. , -0.5,  0. ,  0.5,  1. ])
	print(f'{grid_x.shape=}')
	return grid_x

def generate_subdomain(N_el_x, N_el_y):
	# Construction of RHS for VPINNs
	NE_x, NE_y = N_el_x, N_el_y
	[x_l, x_r] = [-1, 1]
	[y_l, y_u] = [-1, 1]
	delta_x    = (x_r - x_l)/NE_x
	delta_y    = (y_u - y_l)/NE_y
	grid_x     = jnp.asarray([ x_l + i*delta_x for i in range(NE_x+1)])         # array([-1. , -0.5,  0. ,  0.5,  1. ])
	grid_y     = jnp.asarray([ y_l + i*delta_y for i in range(NE_y+1)])         # array([-1. , -0.5,  0. ,  0.5,  1. ])
	print(f'{grid_x.shape=}')
	print(f'{grid_y.shape=}')
	return grid_x, grid_y

def Test_fcn_x(n,x):
	test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
	return test
def Test_fcn_y(n,y):
	test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
	return test

def compute_u_exact_f_external(XY_quad_train, WXY_quad_train, NE_x, NE_y, N_test_x, N_test_y, N_testfcn_total, grid_x, grid_y):
	x_quad  = XY_quad_train[:,0:1]      # (100, 1)
	y_quad  = XY_quad_train[:,1:2]      # (100, 1)
	w_quad  = WXY_quad_train            # (100, 2)
	U_ext_total = []
	F_ext_total = []

	for ex in range(NE_x):
		for ey in range(NE_y):
			Ntest_elementx  = N_testfcn_total[0][ex]
			Ntest_elementy  = N_testfcn_total[1][ey]
			
			x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1)
			y_quad_element = grid_y[ey] + (grid_y[ey+1]-grid_y[ey])/2*(y_quad+1)
			jacobian       = ((grid_x[ex+1]-grid_x[ex])/2)*((grid_y[ey+1]-grid_y[ey])/2)
			
			testx_quad_element = np.asarray([ Test_fcn_x(n,x_quad)  for n in range(1, Ntest_elementx+1)])
			testy_quad_element = np.asarray([ Test_fcn_y(n,y_quad)  for n in range(1, Ntest_elementy+1)])
	
			u_quad_element = u_ext(x_quad_element, y_quad_element)
			f_quad_element = f_ext(x_quad_element, y_quad_element)
			
			U_ext_element = jnp.asarray([[jacobian*np.sum(\
							w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*u_quad_element) \
							for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
	
			F_ext_element = jnp.asarray([[jacobian*np.sum(\
							w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) \
							for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
			
			U_ext_total.append(U_ext_element)
	
			F_ext_total.append(F_ext_element)
	
	F_ext_total = np.reshape(F_ext_total, [NE_x, NE_y, N_test_y[0], N_test_x[0]])
	F_ext_total = jnp.array(F_ext_total)
	print(f'{len(U_ext_total)=}')     # len(U_ext_total)=16
	print(f'{F_ext_total.shape=}')    # F_ext_total.shape=(4, 4, 5, 5)
	return U_ext_total, F_ext_total

def generate_test_set():
	# Test points
	[x_l, x_r] = [-1, 1]
	[y_l, y_u] = [-1, 1]
	delta_test = 0.01
	xtest = np.arange(x_l, x_r + delta_test, delta_test)
	ytest = np.arange(y_l, y_u + delta_test, delta_test)
	data_temp = np.asarray([[ [xtest[i], ytest[j], u_ext(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))])
	Xtest = data_temp.flatten()[0::3]
	Ytest = data_temp.flatten()[1::3]
	Exact = data_temp.flatten()[2::3]
	X_test = np.hstack((Xtest[:,None],Ytest[:,None]))
	u_test = Exact[:,None]
	print(f'{X_test.shape=}')
	print(f'{u_test.shape=}')
	return X_test, u_test, ytest

def SGJW_Test_fn(Ntest, quad):
	test_total = []
	alpha = [1,2,3,1,1]
	beta = [1,1,1,2,4]
	assert len(alpha) == Ntest == len(beta)
	for a,b in zip(alpha, beta):
		local_params = {'alpha':a, 'beta':b, }
		test = standard_Jacobi_weight_fn(quad, local_params)
		test_total.append(test)
	return jnp.asarray(test_total)

def generate_test_fn_and_derivative(XY_quad_train):
	# Compute the test function and their derivatives apriori, as they use jacobi scipy which is not jit compatible
	Nelementx = 5
	Nelementy = 5
	xquad  = jnp.array(XY_quad_train)[:,0:1]
	yquad  = jnp.array(XY_quad_train)[:,1:2]
	all_testx = []
	all_testy = []
	all_d2testx = []
	all_d2testy = []
	for ex in range(Nelementx):
		temp_testx = []
		temp_testy = []
		temp_d2testx = []
		temp_d2testy  = []
		for ey in range(Nelementy):
			Ntest_elementx = 5
			Ntest_elementy = 5
			# testx_quad_element = SGJW_Test_fn(Ntest_elementx, xquad)
			testx_quad_element = Test_fcnx(Ntest_elementx, xquad)
			d1testx_quad_element, d2testx_quad_element = dTest_fcn(Ntest_elementx, xquad)
			# testy_quad_element = SGJW_Test_fn(Ntest_elementy, yquad)
			testy_quad_element = Test_fcny(Ntest_elementy, yquad)
			d1testy_quad_element, d2testy_quad_element = dTest_fcn(Ntest_elementy, yquad)
			
			temp_testx.append(testx_quad_element)
			temp_testy.append(testy_quad_element)
			temp_d2testx.append(d2testx_quad_element)
			temp_d2testy.append(d2testy_quad_element)
		
		all_testx.append(temp_testx)
		all_testy.append(temp_testy)
		all_d2testx.append(temp_d2testx)
		all_d2testy.append(temp_d2testy)
	return all_testx, all_testy, all_d2testx, all_d2testy

def gen_index_1D(nx):
	index_combine_x = []
	for x in range(nx):
		index_combine_x.append(x)
	index_combine_x = jnp.asarray(index_combine_x)
	return index_combine_x

def combine_index(nx, ny):
	index_combine_x = []
	index_combine_y = []
	for x in range(nx):
		for y in range(ny):
			index_combine_x.append(x)
			index_combine_y.append(y)
	index_combine_x = jnp.asarray(index_combine_x)
	index_combine_y = jnp.asarray(index_combine_y)
	return index_combine_x, index_combine_y

def compute_J_zero_beta_value(N_degree, alpha, beta):
	# Follow heuristic, to select left and right edge demarcation
	k_left = math.ceil(math.sqrt(N_degree))
	k_right = N_degree - math.ceil(math.sqrt(N_degree))
	print("Left demarcation:", k_left)
	print("Right demarcation:", k_right)
	# get zeros of the BEssEl function
	J_left_zeros = jn_zeros(beta, k_left)
	all_right_zero = jn_zeros(alpha, k_right)

	num_right_edge = N_degree - k_right
	right_zero_a = all_right_zero[:num_right_edge] 
	right_zero_b = all_right_zero[-num_right_edge:]
	J_right_zeros = right_zero_a

	# evaluate the lower order BEssEl function at the corresponding zero
	J_beta_value_left = jn(beta-1, J_left_zeros)
	J_beta_value_right = jn(alpha-1, J_right_zeros)
	return jnp.array(J_left_zeros), jnp.array(J_right_zeros), jnp.array(J_beta_value_left), jnp.array(J_beta_value_right)

def root(k, mod_model, coeff_model, local_params):
	bisec = Bisection(optimality_fun=F, lower=-1, upper=+1, check_bracket=False)
	solve = bisec.run(k=k, mod_model=mod_model, a=local_params['alpha'], b=local_params['beta'], n=local_params['N'])           # compute root
	# print(k, solve.params, solve.state.value)
	post_solve_value = post_solve(solve.params, mod_model, coeff_model, local_params)
	return post_solve_value, (k, solve.params, solve.state.value)

def F(tk, k, mod_model, a, b, n):
	c1 = 4*k + 2*a + 3  
	c2 = 4*n + 2*a + 2*b + 2
	c3 = 2*n + a + b + 1
	F = (jnp.pi * c1) / c2 - jnp.arccos(tk) - (2*jnp.pi*jnp.log(modulation_net_forward_pass(tk, mod_model)+1e-8))/c3
	return F[0]    # need to return scalar for root finding 

def weight_fn(xk, mod_model, local_params):
	# modified Jacobi weight function
	a = local_params['alpha']
	b = local_params['beta']
	h_xk = modulation_net_forward_pass(xk, mod_model)
	weight_x = jnp.power(1-xk, a)*jnp.power(1+xk, b)*h_xk
	return weight_x

def standard_Jacobi_weight_fn(xk, local_params):
	# standard Jacobi weight function
	a = local_params['alpha']
	b = local_params['beta']
	weight_x = jnp.power(1-xk, a)*jnp.power(1+xk, b)
	return weight_x

def node_xk(leading_tk, h_coeff, c_coeff, d_coeff, local_params):
	a = local_params['alpha']
	b = local_params['beta']
	n = local_params['N']
	term_one = leading_tk
	term_two = ((2*jnp.power(a,2) - 2*jnp.power(b,2) + (2*jnp.power(a,2)+2*jnp.power(b,2)-1) )*(term_one))/(2*jnp.power(2*n+a+b+1+h_coeff[0], 2))
	common_factor = (-1.0)/(4*jnp.power(2*n+a+b+1+h_coeff[0], 3))
	term_three = 2*(2*jnp.power(a,2)+2*jnp.power(b,2)-1)*h_coeff[1]*jnp.power(term_one, 3)
	term_four = 2*((2*jnp.power(a,2)+2*jnp.power(b,2)-1)*h_coeff[0] + 2*(jnp.power(a,2) - jnp.power(b,2))*h_coeff[1])*jnp.power(term_one, 2)
	term_five = (4*jnp.power(a,2)-1)*c_coeff
	term_six = (4*jnp.power(b,2)-1)*d_coeff
	term_seven = 8*(jnp.power(a,2) - jnp.power(b,2))*h_coeff[0] - 4*(jnp.power(a,2) - jnp.power(b,2))*h_coeff[1]
	term_eight = term_five - term_six  + 4*(3*jnp.power(a,2)+jnp.power(b,2)-1)*h_coeff[0]
	term_nine = -2.0*(2*jnp.power(a,2)+2*jnp.power(b,2)-1)*h_coeff[1]
	term_ten = (term_eight + term_nine)*term_one
	term_eleven = common_factor*(term_three + term_four + term_five + term_six + term_seven + term_ten)
	final_xk = term_one + term_two + term_eleven
	return final_xk

def weight_wk(leading_tk, h_coeff, weight_xk, local_params):
	a = local_params['alpha']
	b = local_params['beta']
	n = local_params['N']
	tk = leading_tk
	pre_fac = (jnp.pi*(jnp.sqrt(1 - jnp.power(tk, 2))))/(2*n+a+b+1)
	term_one = 2 - (2*h_coeff[1]*(1 - jnp.power(tk,2)) - 2*h_coeff[0]*tk)/(2*n+a+b+1)
	common_factor = 1/(jnp.power(2*n + a + b + 1, 2))
	term_two = 2*jnp.power(h_coeff[1], 2)*jnp.power(tk, 4) + 4*h_coeff[0]*h_coeff[1]*jnp.power(tk, 3) - 4*h_coeff[0]*h_coeff[1]*tk
	term_three = 2*(jnp.power(h_coeff[0], 2) - 2*jnp.power(h_coeff[1], 2))*jnp.power(tk, 2)
	term_four = 2*jnp.power(a,2) + 2*jnp.power(b,2) + 2*jnp.power(h_coeff[1],2) - 1
	term_five = pre_fac*(term_one + common_factor*(term_two + term_three + term_four))
	final_wk = weight_xk*term_five
	final_wk = jax.nn.relu(final_wk)
	# final_wk = jax.nn.softplus(final_wk)
	return final_wk

def post_solve(root_tk, mod_model, coeff_model, local_params):
	# compute the h, c and d coefficient
	all_coeff = generic_tanh_forward_pass(root_tk, coeff_model)
	
	Ntest_perelem = 5
	all_coeff = all_coeff.reshape(Ntest_perelem, -1)

	h_coeff = all_coeff[:, :2]
	c_coeff = all_coeff[:, 2]
	d_coeff = all_coeff[:, 3]
	vec_node_xk = jax.vmap(node_xk, in_axes=(None, 0, 0, 0, None))
	x_k = vec_node_xk(root_tk, h_coeff, c_coeff, d_coeff, local_params)
	vec_weight_fn = jax.vmap(weight_fn, in_axes=(0, None, None))
	w_of_xk = vec_weight_fn(x_k, mod_model, local_params)
	vec_weight_wk = jax.vmap(weight_wk, in_axes=(None, 0, 0, None)) 
	w_k = vec_weight_wk(root_tk, h_coeff, w_of_xk, local_params)
	return x_k, w_k

def edge_weight_wk(k, c_coeff, d_coeff, weight_xk, local_params, left_edge):
	#compute weight of node
	if left_edge:
		a = local_params['alpha']
		b = local_params['beta']
		n = local_params['N']
		jbk = local_params['JL_zero'][k-1]
		J_value = local_params['JL_beta'][k-1]
	else:
		a = local_params['beta']
		b = local_params['alpha']
		n = local_params['N']
		jbk = local_params['JR_zero'][k-1]
		J_value = local_params['JR_beta'][k-1]
		
	term_one = (8)/(jnp.power(J_value, 2)*(jnp.power(2*n+a+b+1-d_coeff[0], 2)))
	term_two = (8*(3*jnp.power(a, 2) + jnp.power(b, 2) -1 -2*jnp.power(jbk, 2)))/(3*jnp.power(J_value, 2)*(jnp.power(2*n+a+b+1-d_coeff[0], 4)))
	neum = (-2)*( 32*(d_coeff[0] - 3*d_coeff[1])*jnp.power(jbk,2) + 3*(4*jnp.power(a,2) - 1)*c_coeff + (12*a*a + 8*b*b - 5)*d_coeff[0] - 6*(4*b*b - 1)*d_coeff[1] )
	deno = 3*jnp.power(J_value, 2)*(jnp.power(2*n+a+b+1-d_coeff[0], 5))
	term_three = term_one + term_two + (neum/deno)
	final_wk = weight_xk*term_three
	final_wk = jax.nn.relu(final_wk)
	# final_wk = jax.nn.softplus(final_wk)
	return final_wk

def edge_node_xk(k, c_coeff, d_coeff, local_params, left_edge):
	# compute node value
	if left_edge:
		a = local_params['alpha']
		b = local_params['beta']
		n = local_params['N']
		jbk = local_params['JL_zero'][k-1]
	else:
		a = local_params['beta']
		b = local_params['alpha']
		n = local_params['N']
		jbk = local_params['JR_zero'][k-1]

	term_one = (2*(jnp.power(jbk, 2)))/(jnp.power(2*n+a+b+1-d_coeff[0], 2))
	term_two = ((-2)*(jnp.power(jbk, 2))*(jnp.power(jbk, 2) - 3*jnp.power(a, 2) - jnp.power(b, 2) + 1 ))/(3*jnp.power(2*n+a+b+1-d_coeff[0], 4))
	common_factor = ((-1)*(jnp.power(jbk, 2)))/(6*jnp.power(2*n+a+b+1-d_coeff[0], 5))
	term_three = 16*(d_coeff[0] - 3*d_coeff[1])*jnp.power(jbk, 4) + 3*(4*jnp.power(a,2)-1)*c_coeff
	term_four = (12*jnp.power(a, 2) + 8*jnp.power(b,2) - 5)*d_coeff[0] - 6*(4*jnp.power(b,2) - 1)*d_coeff[1]
	final_xk = -1 + term_one + term_two  + common_factor*(term_three + term_four)
	if left_edge:
		final_xk = final_xk
	else:
		final_xk = -1*final_xk
	return final_xk


def edge_compute(k, problem_specific, modulation_model, edge_model, local_params):
	# compute coeff c0 and d0, d1 for the edge asymptote
	all_coeff = generic_tanh_forward_pass(problem_specific, edge_model)

	Ntest_perelem = 5
	all_coeff = all_coeff.reshape(Ntest_perelem, -1)
	left_c = all_coeff[:, 0]
	left_d = all_coeff[:, 1:3]
	right_c = all_coeff[:, 3]
	right_d = all_coeff[:, 4:]

	vec_edge_node_xk = jax.vmap(edge_node_xk, in_axes=(None, 0, 0, None, None))
	vec_weight_fn = jax.vmap(weight_fn, in_axes=(0, None, None))
	vec_edge_weight_wk = jax.vmap(edge_weight_wk, in_axes=(None, 0, 0, 0, None, None))

	l_x_k = vec_edge_node_xk(k, left_c,  left_d, local_params, True)[:,0]
	l_w_of_xk = vec_weight_fn(l_x_k, modulation_model, local_params)
	l_w_k = vec_edge_weight_wk(k, left_c, left_d, l_w_of_xk, local_params, True)
	l_x_k = jax.lax.expand_dims(l_x_k, [-1])

	r_x_k = vec_edge_node_xk(k, right_c,  right_d, local_params, False)[:,0]
	r_w_of_xk = vec_weight_fn(r_x_k, modulation_model, local_params)
	r_w_k = vec_edge_weight_wk(k, right_c, right_d, r_w_of_xk, local_params, False)
	r_x_k = jax.lax.expand_dims(r_x_k, [-1])
	return l_x_k, l_w_k, r_x_k, r_w_k

def tensor_prod(x_node, y_node, x_weight, y_weight):
	x_grid, y_grid = jnp.meshgrid(x_node, y_node)
	x_flat = jnp.ravel(x_grid)
	y_flat = jnp.ravel(y_grid)
	node_xk = jnp.stack((x_flat, y_flat), axis=1)

	wx_grid, wy_grid = jnp.meshgrid(x_weight, y_weight)	
	wx_flat = jnp.ravel(wx_grid)
	wy_flat = jnp.ravel(wy_grid)
	node_wk = jnp.stack((wx_flat, wy_flat), axis=1)
	weight = jnp.multiply(node_wk[:,0], node_wk[:,1])
	
	return node_xk, node_wk, weight




