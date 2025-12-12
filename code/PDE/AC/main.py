import equinox as eqx
import jax
from scipy.io import loadmat

import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import pdb
import math
import os

from jax import random
from jax import config
# config.update("jax_enable_x64", True)
import numpy as onp
from pyDOE import lhs
import time as sys_time
import pickle
from scipy import integrate
from functools import partial
from jax import jit

import sys
sys.path.append('../../')
from GaussJacobi import GaussLobattoJacobiWeights, Jacobi
from util_learn_quad import initialize_mlp_xavier, initialize_mlp_he, initialize_mlp_scale, u_ext, plot3d
from util_learn_quad import compute_J_zero_beta_value, weight_fn, unpack_params, plot_save_loss_dict
from util_learn_quad import root, family_root, edge_compute, family_edge_compute
from util_learn_quad import tensor_prod_node

import matplotlib.pyplot as plt
import argparse
import csv

from model import family_forward_pass, modulation_net_forward_pass
from model import pdesolution_forward_pass
from model import solution_net_forward_pass_v2
import time as sys_time

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for Asymptote in PDE Diffusion Equation")
	parser.add_argument('--seed', type=int, default=52, help="Setting seed for the entire experiment")
	parser.add_argument('--exp', default='New_ac_single_vdeg_dirsol_stime_colstack', help="Adjusted in code: Experiment foler name")
	
	parser.add_argument('--NEpoch', type=int, default=200000, help='Number of training epoch for stage 1')
	parser.add_argument('--data_size', type=int, default=100, help='Total size of train plus test set')
	parser.add_argument('--minval', default=0.01, type=float, help="Minimum value for uniform distribution of parameter")
	parser.add_argument('--maxval', default=0.8, type=float, help="Maximum value for uniform distribution of parameter")
	parser.add_argument('--minV', default=0.01, type=float, help="Minimum value for temperature at boundary")
	parser.add_argument('--maxV', default=0.8, type=float, help="Maximum value for temperature at boundary")
	parser.add_argument('--tolerance', default=1e-5, type=float, help="Tolerance level for integration approximation error")

	parser.add_argument('--penalty', type=float, default=1, help='Weight for loss term due to boundary condition')
	parser.add_argument('--lr', default=0.0001, type=float, help="Learning rate for the asympttoe model")
	parser.add_argument('--decay_step', type=int, default=50, help='Decay for learning rate')
	parser.add_argument('--wd', default=0.00001, type=float, help="L2 regularize for the asymptote model")

	parser.add_argument('--N_degree', type=int, default=1025, help='Degree of polynomial in one direction, which will give asymptotic node and weight')
	parser.add_argument('--min_degree', type=int, default=975, help='Minimum Degree of polynomial in one direction, used in varying case')
	parser.add_argument('--alpha', type=int, default=1, help='alpha value for modified Jacobi polynomial')
	parser.add_argument('--beta', type=int, default=1, help='beta value for modified Jacobi polynomial')

	# argument for model architecture
	parser.add_argument('--family_width', type=int, default=100, help='Per layer width of modulation model')
	parser.add_argument('--family_depth', type=int, default=4, help='Depth of modulation model')

	parser.add_argument('--modulation_width', type=int, default=10, help='Per layer width of modulation function which will be predicted')
	parser.add_argument('--modulation_depth', type=int, default=3, help='Depth of modulation function which will be predicted')
	parser.add_argument('--solve_width', type=int, default=64, help='Per layer width of solution function which will be predicted')
	parser.add_argument('--solve_depth', type=int, default=3, help='Depth of solution function which will be predicted')

	# Saving argument
	parser.add_argument('--print_freq', default=30, type=int, help='Frequency of printing the loss value to temrinal, note this is time consuming in jax')
	parser.add_argument('--plot_loss_freq', default=1000, type=int, help='Frequency to save the loss plot, this is also time consuming in jax')
	parser.add_argument('--model_save_freq', default=1000, type=int, help='Saving frequency of model prediction')
			
	args = parser.parse_args()
	return args

def compute_param_num(size):
	total = 0
	for m,n in zip(size[:-1], size[1:]):
		in_dim = m
		out_dim = n
		total += n + n*m
	return total

def init_distribution(x):
	value = x*x*jnp.cos(jnp.pi*x)
	return value

def right_boundary(t, t2):
	return t2

def left_boundary(t, t1):
	return t1


def inhomogene(x,t):
	val1 = -1*jnp.sin(jnp.pi*x) + jnp.pi*jnp.pi*jnp.sin(jnp.pi*x)
	value = jnp.exp(-t)*val1
	return value

def compute_integral(mod_model, coeff, edge, train_data):
	params = {'N': train_data['degree'], 'alpha': train_data['alpha'], 'beta': train_data['beta'],
			 "JL_zero": train_data['JL_zero'], "JR_zero": train_data['JR_zero'], "JL_beta": train_data['JL_beta'], "JR_beta": train_data['JR_beta']}
	
	# quadrature position and weight
	K_bulk = train_data['K_bulk']
	vec_root = jax.vmap(family_root, in_axes=(0, None, None, None))
	(bulk_x_node, bulk_x_weight) = vec_root(K_bulk, mod_model, coeff, params)
	bulk_x_node = jnp.expand_dims(bulk_x_node, [-1])
	
	K_edge = train_data['K_edge']
	vec_edge_compute = jax.vmap(family_edge_compute, in_axes=(0, None, None, None))
	left_x_node, left_x_weight, right_x_node, right_x_weight = vec_edge_compute(K_edge, mod_model, edge, params)
	left_x_node = jnp.flip(left_x_node, axis=0)

	x_node = jax.lax.concatenate((right_x_node, bulk_x_node, left_x_node), dimension=0)[:,0,0]
	x_weight = jax.lax.concatenate((right_x_weight, bulk_x_weight, left_x_weight), dimension=0)[:,0,0]

	vec_weight_fn = jax.vmap(weight_fn, in_axes=(0, None, None))
	weight_eval = vec_weight_fn(train_data['pre_quad_pt'], mod_model, params).squeeze(-1)
	des_sum = jnp.sum(train_data['pre_quad_wt']*weight_eval)
	penalty_des_sum = jnp.mean((des_sum - 2)**2)
	curr_sum = jnp.sum(x_weight)
	wloss_sum = (curr_sum - des_sum)**2
	return x_node, x_weight, penalty_des_sum, wloss_sum

@partial(jit, static_argnums=(4,5,6,7,8,9))
def loss(t_model, fam_model, sol_model, train_data, mod_num, mod_width, mod_depth, sol_width, sol_depth, bc_penalty):
	vec_family = jax.vmap(family_forward_pass, in_axes=(0, None))
	X = jnp.expand_dims(train_data['X'], [0])

	fwd_pass = vec_family(X, fam_model).squeeze()	
	modulation = fwd_pass[:mod_num]
	domain_coeff = fwd_pass[mod_num:mod_num+4]
	edge_coeff = fwd_pass[mod_num+4:]
	size = [1] + [mod_width]*mod_depth + [1]
	modulator = unpack_params(size, modulation)
	x_node, x_weight, penalty_des_sum, wloss_sum = compute_integral(modulator, domain_coeff, edge_coeff, train_data)
	x_node_lin = x_node

	fwd_pass = vec_family(X, t_model).squeeze()	
	modulation = fwd_pass[:mod_num]
	domain_coeff = fwd_pass[mod_num:mod_num+4]
	edge_coeff = fwd_pass[mod_num+4:]
	size = [1] + [mod_width]*mod_depth + [1]
	modulator = unpack_params(size, modulation)
	time, t_weight, t_penalty_des_sum, t_wloss_sum = compute_integral(modulator, domain_coeff, edge_coeff, train_data)
	
	time = (time+1)/2     # bring in 0 to 1
	time_lin = time
	time = jax.random.permutation(train_data['key'], time) 

	combine_node = jnp.column_stack((x_node, time))
	x_node = combine_node[:,0]
	time = combine_node[:,1]

	solver = sol_model

	dU_dx = jax.grad(solution_net_forward_pass_v2, argnums=0, has_aux=False)
	d2U_dx2 = jax.grad(dU_dx, argnums=0, has_aux=False)				
	d2U_dx2_vec = jax.vmap(d2U_dx2, in_axes=(0, 0, None))
	d2ux_value = d2U_dx2_vec(x_node.squeeze(), time.squeeze(), solver)

	dU_dt = jax.grad(solution_net_forward_pass_v2, argnums=1, has_aux=False)
	dU_dt_vec = jax.vmap(dU_dt, in_axes=(0, 0, None))
	dut_value = dU_dt_vec(x_node.squeeze(), time.squeeze(), solver)
	
	U_vec = jax.vmap(solution_net_forward_pass_v2, in_axes=(0, 0, None))
	Uvalue = U_vec(x_node.squeeze(), time.squeeze(), solver)
	Uvalue3=Uvalue*Uvalue*Uvalue
	inhomogeneous_term = 5*(Uvalue3-Uvalue)

	diffu = train_data['X'][-1]
	domain_loss = jnp.mean((dut_value - diffu*d2ux_value + inhomogeneous_term)**2)

	init_dist_pred = jax.vmap(solution_net_forward_pass_v2, in_axes=(0, None, None))(x_node_lin.squeeze(), 0.0, solver)
	init_dist_true = jax.vmap(init_distribution, in_axes=(0))(x_node_lin.squeeze())
	init_dist_lose = jnp.mean((init_dist_pred - init_dist_true)**2)

	left_temp_pred = jax.vmap(solution_net_forward_pass_v2, in_axes=(None, 0, None))(-1.0, time_lin.squeeze(), solver)
	left_temp_true = jax.vmap(left_boundary, in_axes=(0, None))(time_lin.squeeze(), -1)
	left_bc_err = jnp.mean((left_temp_pred - left_temp_true)**2)

	right_temp_pred = jax.vmap(solution_net_forward_pass_v2, in_axes=(None, 0, None))(1.0, time_lin.squeeze(), solver)
	right_temp_true = jax.vmap(right_boundary, in_axes=(0, None))(time_lin.squeeze(), -1)
	right_bc_err = jnp.mean((right_temp_pred - right_temp_true)**2)

	ic_loss = init_dist_lose + left_bc_err + right_bc_err
	loss = domain_loss + bc_penalty*ic_loss + penalty_des_sum + wloss_sum + t_penalty_des_sum + t_wloss_sum
	loss_dict = {'total':loss, 'domain':domain_loss, 'ic':ic_loss, 'we_penalty':penalty_des_sum, 'we_sum':wloss_sum}
	return loss, (loss_dict, solver)

def l2_relative_error(y_true, y_pred):
	return jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

def gen_testdata():
	data = loadmat("usol_D_0.001_k_5.mat")
	t = data["t"]
	x = data["x"]
	u = data["u"]
	import numpy as naivepy
	xx, tt = naivepy.meshgrid(x, t)
	X = naivepy.vstack((naivepy.ravel(xx), naivepy.ravel(tt))).T
	y = u.flatten()[:, None]
	return jnp.asarray(X), jnp.asarray(y)

def main():
	print("Check validity of asymptote quadrature, 1D wave equation")
	args = parse_args()
	arg_dict = vars(args)
	exp_name = args.exp
	print(args)
	if os.path.exists('./Result/'+exp_name):
		nth_exp = len(os.listdir('./Result/'+exp_name+'/Results'))+1
	else:
		nth_exp = 0
	args.exp = './Result/'+exp_name+'/Results/'+str(nth_exp)
	arg_dict['Result_location'] = './Results/'+str(nth_exp)
	if not os.path.exists(args.exp):
		print("Creating experiment directory: ", args.exp)
		os.makedirs(args.exp)
		os.makedirs(args.exp+'/quad/')
		os.makedirs(args.exp+'/sol/')
		os.makedirs(args.exp+'/train_sol/')
		os.makedirs(args.exp+'/test_sol/')
		os.makedirs(args.exp+'/data/')
		os.makedirs(args.exp+'/model/')
		dir_name = args.exp
	else:
		print("REsult path already exits, please check!!")
		exit(0)

	# list of column names 
	field_names = arg_dict.keys()
	argument_storage_file = './Result/'+exp_name+'/experiment.csv'
	if os.path.exists(argument_storage_file):
		with open(argument_storage_file, 'a') as csv_file:
			dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
			dict_object.writerow(arg_dict)
	else:
		print("First run of experiment, so create new storage for hyperparameter")
		with open(argument_storage_file, 'w') as csv_file:
			writer = csv.writer(csv_file) 
			writer.writerow(arg_dict.keys()) 
			writer.writerow(arg_dict.values())
	print(arg_dict)
	argument_file = args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)
		
	key = random.PRNGKey(args.seed)
	onp.random.seed(args.seed)
	N_degree=args.N_degree
	alpha = args.alpha
	beta = args.beta

	solution_key, time_key, family_key, prev_epoch_key = random.split(key, 4)
	
	modulation_num = compute_param_num([1] + [args.modulation_width]*args.modulation_depth + [1])
	family_model_size = [1] + [args.family_width]*args.family_depth + [modulation_num+6+4]

	family_model = initialize_mlp_xavier(family_model_size, family_key)
	# fam_lr_schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.decay_step, alpha=0.0, exponent=1.0)
	# family_optim = optax.adam(learning_rate=fam_lr_schedule)
	family_optim = optax.adam(learning_rate=args.lr)
	family_state = family_optim.init(family_model)
	
	time_model = initialize_mlp_xavier(family_model_size, time_key)
	# time_lr_schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.decay_step, alpha=0.0, exponent=1.0)
	# time_optim = optax.adam(learning_rate=time_lr_schedule)
	time_optim = optax.adam(learning_rate=args.lr)
	time_state = time_optim.init(time_model)

	solution_model_size = [2] + [args.solve_width]*args.solve_depth + [1]
	solution_model = initialize_mlp_xavier(solution_model_size, solution_key)
	# sol_lr_schedule = optax.cosine_decay_schedule(init_value=args.lr, decay_steps=args.decay_step, alpha=0.0, exponent=1.0)
	# solution_optim = optax.adam(learning_rate=sol_lr_schedule)
	solution_optim = optax.adam(learning_rate=args.lr)
	solution_state = solution_optim.init(solution_model)

	mod_num, mod_width, mod_depth, sol_width, sol_depth = modulation_num, args.modulation_width, args.modulation_depth, args.solve_width, args.solve_depth
	bc_penalty = args.penalty

	@jax.jit
	def joint_update(t_model, t_state, fam_model, fam_state, sol_model, sol_state, train_data):
		loss_fn_grad_value = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)
		((loss_value, (loss_dict, solve)), gradient) = loss_fn_grad_value(t_model, fam_model, sol_model, train_data, mod_num, mod_width, mod_depth, sol_width, sol_depth, bc_penalty)
		updates, t_state = time_optim.update(gradient[0], t_state, t_model)
		t_model = optax.apply_updates(t_model, updates)
		updates, fam_state = family_optim.update(gradient[1], fam_state, fam_model)
		fam_model = optax.apply_updates(fam_model, updates)
		updates, sol_state = solution_optim.update(gradient[2], sol_state, sol_model)
		sol_model = optax.apply_updates(sol_model, updates)
		return t_model, t_state, fam_model, fam_state, sol_model, sol_state, solve, loss_value, loss_dict
	
	mu=0.001
	test_x, test_y = gen_testdata()
	pre_quad_pt, pre_quad_wt  = GaussLobattoJacobiWeights(100, 0, 0)
	pre_quad_pt, pre_quad_wt = jnp.asarray(pre_quad_pt), jnp.asarray(pre_quad_wt)
	
	print("Pre-computing constant data")
	full_data=[]
	for i in range(args.min_degree, args.N_degree):
		N_degree=i
		J_left_zeros, J_right_zeros, J_beta_value_left, J_beta_value_right = compute_J_zero_beta_value(N_degree, alpha, beta)
		K_bulk = 1.0*jnp.arange(math.ceil(math.sqrt(N_degree))+1, N_degree-math.ceil(math.sqrt(N_degree))+1)
		K_bulk = jnp.expand_dims(K_bulk, [-1])
		K_edge = jnp.arange(1, math.ceil(math.sqrt(N_degree))+1)
		K_edge = jnp.expand_dims(K_edge, [-1])
		train_data_i = {"JL_zero": J_left_zeros, "JR_zero": J_right_zeros, "JL_beta": J_beta_value_left, "JR_beta": J_beta_value_right,
					"K_bulk": K_bulk, "K_edge": K_edge, "degree": N_degree, "alpha": alpha, "beta": beta,
					"pre_quad_pt": pre_quad_pt, "pre_quad_wt": pre_quad_wt, 'X': jnp.array([mu])}
		full_data.append(train_data_i)
	len_data=len(full_data)
	
	print("Start PDE solution training")
	start_time = sys_time.time()
	all_loss = {'train': [], 'test': []}
	min_l2_error=1000
	min_epoch=None

	for epoch in range(args.NEpoch):

		prev_epoch_key, data_key, N_key = random.split(prev_epoch_key, 3)
		data_i = int(jax.random.randint(N_key, minval=0, maxval=len_data, shape=(1,))[0])
		train_data=full_data[data_i]
		train_data['key'] = data_key

		time_model, time_state, family_model, family_state, solution_model, solution_state, solve, loss_value, loss_dict = joint_update(time_model, time_state, family_model, family_state, solution_model, solution_state, train_data)
		if math.isnan(loss_value):
			print("Encountered NaN value in loss")
			print("Train Epoch: {}, Loss:{}, Loss_dict:{}".format(epoch, loss_value, loss_dict))
			pdb.set_trace()
		print(loss_dict)
		train_loss = loss_value
		all_loss['train'].append(train_loss)

		y_pred = jax.vmap(solution_net_forward_pass_v2, in_axes=(0, 0, None))(test_x[:,0], test_x[:,1], solve)
		test_loss = l2_relative_error(test_y.squeeze(), y_pred)
		all_loss['test'].append(test_loss)

		print("###Epoch: {}, Train Loss:{}, Test Loss:{}".format(epoch, train_loss, test_loss))
		if min_l2_error>test_loss:
			min_l2_error=test_loss
			min_epoch=epoch
			print("L2 relative error:", min_l2_error)

		if epoch%args.plot_loss_freq==0:
			plot_save_loss_dict(all_loss, epoch, dir_name, filename='solver_losse.png', plot_separate=False)
			print("###Epoch: {}, Train Loss:{}, Test Loss:{}".format(epoch, train_loss, test_loss))
			print("L2 relative error:", min_l2_error)
			print("L2 min epoch:", min_epoch)

	print("L2 relative error:", min_l2_error)
	print("L2 min epoch:", min_epoch)

	print(sys_time.time()-start_time)
	plot_save_loss_dict(all_loss, epoch, dir_name, filename='solver_losse.png', plot_separate=False)
	file_name = args.exp+'/model/'+'final_model.npy'
	with open(file_name, 'wb') as file:
		pickle.dump(family_model, file)
	file_name = args.exp+'/model/'+'solution_model.npy'
	with open(file_name, 'wb') as file:
		pickle.dump(solution_model, file)
	print("##############################")
	
if __name__ == "__main__":
	main()