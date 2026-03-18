

import equinox as eqx
import jax

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
from util_learn_quad import tensor_prod_node, quantize_points

import matplotlib.pyplot as plt
import argparse
import csv

from model import family_forward_pass, modulation_net_forward_pass
from model import pdesolution_forward_pass
from model import solution_net_forward_pass_v2

GAMMA = 2.0
BX = 0.75

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 0.2

ρ_L, u_L, v_L, w_L, By_L, Bz_L, p_L = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0
ρ_R, u_R, v_R, w_R, By_R, Bz_R, p_R = 0.125, 0.0, 0.0, 0.0, -1.0, 0.0, 0.1

IC_SHARPNESS = 60.0

def _prim_to_cons(rho, u, v, w, By, Bz, p):
	E = p / (GAMMA - 1.0) + 0.5 * rho * (u**2 + v**2 + w**2) + 0.5 * (BX**2 + By**2 + Bz**2)
	return jnp.array([rho, rho*u, rho*v, rho*w, By, Bz, E])

U_LEFT  = _prim_to_cons(ρ_L, u_L, v_L, w_L, By_L, Bz_L, p_L)
U_RIGHT = _prim_to_cons(ρ_R, u_R, v_R, w_R, By_R, Bz_R, p_R)

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for MHD 1D VPINN with Learnable Quadrature")
	parser.add_argument('--seed', type=int, default=42, help="Setting seed for the entire experiment")
	parser.add_argument('--exp', default='mhd1d_vpinn', help="Adjusted in code: Experiment folder name")
	
	parser.add_argument('--NEpoch', type=int, default=100000, help='Number of training epoch for stage 1')
	parser.add_argument('--data_size', type=int, default=100, help='Total size of train plus test set')
	parser.add_argument('--minval', default=0.01, type=float, help="Minimum value for uniform distribution of parameter")
	parser.add_argument('--maxval', default=0.8, type=float, help="Maximum value for uniform distribution of parameter")
	parser.add_argument('--minV', default=0.01, type=float, help="Minimum value for temperature at boundary")
	parser.add_argument('--maxV', default=0.8, type=float, help="Maximum value for temperature at boundary")
	parser.add_argument('--tolerance', default=1e-5, type=float, help="Tolerance level for integration approximation error")
	
	parser.add_argument('--penalty', type=int, default=100, help='Weight for loss term due to boundary condition')
	parser.add_argument('--lr', default=0.001, type=float, help="Learning rate for the asymptote model")
	parser.add_argument('--decay_step', type=int, default=50, help='Decay for learning rate')
	parser.add_argument('--wd', default=0.00001, type=float, help="L2 regularize for the asymptote model")
	
	parser.add_argument('--N_degree', type=int, default=100, help='Degree of polynomial in one direction, which will give asymptotic node and weight')
	parser.add_argument('--min_degree', type=int, default=70, help='Minimum Degree of polynomial in one direction, used in varying case')
	parser.add_argument('--alpha', type=int, default=1, help='alpha value for modified Jacobi polynomial')
	parser.add_argument('--beta', type=int, default=1, help='beta value for modified Jacobi polynomial')
	
	# argument for model architecture
	parser.add_argument('--family_width', type=int, default=100, help='Per layer width of modulation model')
	parser.add_argument('--family_depth', type=int, default=4, help='Depth of modulation model')
	
	parser.add_argument('--modulation_width', type=int, default=10, help='Per layer width of modulation function which will be predicted')
	parser.add_argument('--modulation_depth', type=int, default=3, help='Depth of modulation function which will be predicted')
	parser.add_argument('--solve_width', type=int, default=64, help='Per layer width of solution function which will be predicted')
	parser.add_argument('--solve_depth', type=int, default=4, help='Depth of solution function which will be predicted')
	
	# Saving argument
	parser.add_argument('--print_freq', default=500, type=int, help='Frequency of printing the loss value to terminal, note this is time consuming in jax')
	parser.add_argument('--plot_loss_freq', default=5000, type=int, help='Frequency to save the loss plot, this is also time consuming in jax')
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
	s = 0.5 * (1.0 + jnp.tanh(IC_SHARPNESS * x))
	return (1.0 - s) * U_LEFT + s * U_RIGHT

def right_boundary(t, t2):
	return t2

def left_boundary(t, t1):
	return t1

def compute_cfl(u_max, dt, dx, nu):
	value = (u_max*dt)/dx + (2*nu*dt)/(dx*dx)         # value for Burger <=1
	return value

# Compute flux vector F(U); page 406 in Brio & Wu (1988)
def compute_fluxes(U, Bx=BX):
	ρ, ρu, ρv, ρw, By, Bz, E = U
	u = ρu / ρ
	v = ρv / ρ
	w = ρw / ρ
	P = (GAMMA - 1.0) * (E - 0.5*ρ*(u**2 + v**2 + w**2) - 0.5*(Bx**2 + By**2 + Bz**2))
	P_star = P + 0.5*(Bx**2 + By**2 + Bz**2)
	return jnp.array([
		ρu,
		ρu*u + P_star,
		ρv*u - Bx*By,
		ρw*u - Bx*Bz,
		By*u - Bx*v,
		Bz*u - Bx*w,
		(E + P_star)*u - Bx*(Bx*u + By*v + Bz*w),
	])

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

@partial(jit, static_argnums=(4,5,6,7))
def loss(t_model, fam_model, sol_model, train_data, modulation_num, mod_width, mod_depth, bc_penalty):
	mod_size = [1] + [mod_width]*mod_depth + [1]
	fwd_x = family_forward_pass(jnp.array([0.0]), fam_model)
	mod_x = unpack_params(mod_size, fwd_x[:modulation_num])
	x_node, _, pen_x, wloss_x = compute_integral(mod_x, fwd_x[modulation_num:modulation_num+4], fwd_x[modulation_num+4:], train_data)
	
	fwd_t = family_forward_pass(jnp.array([0.0]), t_model)
	mod_t = unpack_params(mod_size, fwd_t[:modulation_num])
	t_node_raw, _, pen_t, wloss_t = compute_integral(mod_t, fwd_t[modulation_num:modulation_num+4], fwd_t[modulation_num+4:], train_data)
	t_node = (t_node_raw + 1.0) * 0.5 * T_MAX
	
	t_node_perm = jax.random.permutation(train_data['key'], t_node)
	
	residuals = jax.vmap(lambda x, t: (
		jax.jacobian(lambda t_: pdesolution_forward_pass(jnp.array([x, t_]), sol_model))(t) +
		jax.jacobian(lambda x_: compute_fluxes(pdesolution_forward_pass(jnp.array([x_, t]), sol_model)))(x)
	))(x_node, t_node_perm)
	loss_pde = jnp.mean(residuals**2)
	
	U_pred_ic = jax.vmap(lambda x: pdesolution_forward_pass(jnp.array([x, 0.0]), sol_model))(x_node)
	U_true_ic = jax.vmap(init_distribution)(x_node)
	loss_ic = jnp.mean((U_pred_ic - U_true_ic)**2)
	
	U_left_pred  = jax.vmap(lambda t: pdesolution_forward_pass(jnp.array([X_MIN, t]), sol_model))(t_node)
	U_right_pred = jax.vmap(lambda t: pdesolution_forward_pass(jnp.array([X_MAX, t]), sol_model))(t_node)
	left_bc_err  = jnp.mean((U_left_pred  - jax.vmap(left_boundary,  in_axes=(0, None))(t_node, U_LEFT))**2)
	right_bc_err = jnp.mean((U_right_pred - jax.vmap(right_boundary, in_axes=(0, None))(t_node, U_RIGHT))**2)
	loss_bc = left_bc_err + right_bc_err
	
	loss = loss_pde + bc_penalty * (loss_ic + loss_bc) + pen_x + wloss_x + pen_t + wloss_t
	loss_dict = {'total':loss, 'pde':loss_pde, 'ic':loss_ic, 'bc':loss_bc,
		'pen_x':pen_x, 'wloss_x':wloss_x, 'pen_t':pen_t, 'wloss_t':wloss_t}
	return loss, (loss_dict, sol_model)

def l2_relative_error(y_true, y_pred):
	return jnp.linalg.norm(y_true - y_pred) / jnp.linalg.norm(y_true)

def gen_testdata():
	X = jnp.load('test_x.npy')
	y = jnp.load('test_y.npy')
	return X, y

def main():
	print("Check validity of asymptote quadrature, 1D MHD equation")
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
	alpha = args.alpha
	beta = args.beta
	N_degree=args.N_degree
	
	solution_key, time_key, family_key, prev_epoch_key = random.split(key, 4)
	
	modulation_num = compute_param_num([1] + [args.modulation_width]*args.modulation_depth + [1])
	family_model_size = [1] + [args.family_width]*args.family_depth + [modulation_num+4+6]
	
	family_model = initialize_mlp_xavier(family_model_size, family_key)
	family_optim = optax.adam(learning_rate=args.lr)
	family_state = family_optim.init(family_model)
	
	time_model = initialize_mlp_xavier(family_model_size, time_key)
	time_optim = optax.adam(learning_rate=args.lr)
	time_state = time_optim.init(time_model)
	
	solution_model_size = [2] + [args.solve_width]*args.solve_depth + [7]
	solution_model = initialize_mlp_xavier(solution_model_size, solution_key)
	solution_optim = optax.adam(learning_rate=args.lr)
	solution_state = solution_optim.init(solution_model)
	
	mod_num, mod_width, mod_depth = modulation_num, args.modulation_width, args.modulation_depth
	bc_penalty = args.penalty
	
	@jax.jit
	def joint_update(t_model, t_state, fam_model, fam_state, sol_model, sol_state, train_data):
		loss_fn_grad_value = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)
		((loss_val, (loss_dict, solve)), grads) = loss_fn_grad_value(t_model, fam_model, sol_model, train_data, modulation_num, mod_width, mod_depth, bc_penalty)
		updates, t_state = time_optim.update(grads[0], t_state, t_model)
		t_model = optax.apply_updates(t_model, updates)
		updates, fam_state = family_optim.update(grads[1], fam_state, fam_model)
		fam_model = optax.apply_updates(fam_model, updates)
		updates, sol_state = solution_optim.update(grads[2], sol_state, sol_model)
		sol_model = optax.apply_updates(sol_model, updates)
		return t_model, t_state, fam_model, fam_state, sol_model, sol_state, solve, loss_val, loss_dict
	
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
					"pre_quad_pt": pre_quad_pt, "pre_quad_wt": pre_quad_wt}
		full_data.append(train_data_i)
	len_data=len(full_data)
	
	print("Start PDE solution training")
	start_time = sys_time.time()
	all_loss = {'train': [], 'test': []}
	min_l2_error=1000
	min_epoch=None
	
	for epoch in range(args.NEpoch):
		prev_epoch_key, N_key, pair_key = random.split(prev_epoch_key, 3)
		data_i = int(jax.random.randint(N_key, minval=0, maxval=len_data, shape=(1,))[0])
		train_data=full_data[data_i]
		train_data['key'] = pair_key
	
		time_model, time_state, family_model, family_state, solution_model, solution_state, solve, loss_value, loss_dict = joint_update(
			time_model, time_state, family_model, family_state, solution_model, solution_state, train_data)
	
		if math.isnan(loss_value):
			print("Encountered NaN value in loss")
			print("Train Epoch: {}, Loss:{}, Loss_dict:{}".format(epoch, loss_value, loss_dict))
			break
	
		train_loss = loss_value
		all_loss['train'].append(train_loss)
	
		y_pred = jax.vmap(lambda xt: pdesolution_forward_pass(xt, solve))(test_x)
		test_loss = l2_relative_error(test_y, y_pred)
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
	
		# if epoch%args.model_save_freq==0:
		# 	print("Saving model")
		# 	file_name = args.exp+'/model/'+'family_model_'+str(epoch)+'.npy'
		# 	with open(file_name, 'wb') as file:
		# 		pickle.dump(family_model, file)
		# 	file_name = args.exp+'/model/'+'solution_model_'+str(epoch)+'.npy'
		# 	with open(file_name, 'wb') as file:
		# 		pickle.dump(solution_model, file)
	
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
