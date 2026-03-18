# PINNs with Learnable Quadrature

Code for **"PINNs with Learnable Quadrature"** — NeurIPS 2025

[Paper](https://openreview.net/pdf?id=Yg5QkFNHnT) ||
[Video](https://recorder-v3.slideslive.com/#/share?share=106317&s=a1eaa201-733f-4467-b5ca-5d2cb66c1a1b) ||
[Slides](https://neurips.cc/media/neurips-2025/Slides/117404.pdf)

---

## Overview

Standard PINNs evaluate the PDE residual at fixed collocation points (e.g. random or Latin hypercube samples), which can be inefficient and biased toward easy regions of the domain. This work introduces **LearnQuad**: a framework that jointly learns a problem-adapted quadrature rule alongside the PINN solution network.

The key idea is to parameterize the quadrature weight function as:

```
w_θ(x) = (1 - x)^α (1 + x)^β h_θ(x),   x ∈ [-1, 1]
```

where `h_θ` is a small neural network. This induces a family of orthogonal polynomials whose roots serve as quadrature nodes and whose associated weights are computed via asymptotic expansions. The quadrature rule and the PDE solution network are trained end-to-end, with a regularization loss that keeps the rule well-normalized.

---

## Repository Structure

```
learn-quad/
├── README.md
├── req.txt                        # Full dependency list
├── LICENSE
└── code/
    ├── GaussJacobi.py             # Gauss-Jacobi and Gauss-Lobatto quadrature rules
    ├── model.py                   # Neural network forward passes and architectures
    ├── util.py                    # General utilities
    ├── util_learn_quad.py         # Core LearnQuad library (asymptotic expansions,
    │                              #   quadrature node/weight computation, MLP init)
    └── PDE/
        ├── AC/main.py             # Allen-Cahn equation
        ├── burger/main.py         # Burgers equation
        ├── convection/main.py     # Convection equation
        ├── diffusion/main.py      # Diffusion equation
        ├── wave/main.py           # Wave equation
        └── mhd/Main.py            # 1D ideal MHD equations (Brio-Wu Riemann problem)
```

---

## File Descriptions

### `code/GaussJacobi.py`
Implements standard Gauss-Jacobi and Gauss-Lobatto-Jacobi quadrature rules. Used to pre-compute a fixed high-accuracy reference rule (`GaussLobattoJacobiWeights`) that evaluates the integral of the learned weight function during training.

### `code/model.py`
Contains all neural network forward passes:
- `family_forward_pass` — forward pass for the family/modulation network that outputs quadrature parameters
- `pdesolution_forward_pass` — forward pass for the PDE solution network (tanh MLP, vector output)
- `solution_net_forward_pass_v2` — scalar-output solution network used by the 1D PDE experiments
- `modulation_net_forward_pass` — forward pass for the small modulation network with softplus output
- Equinox-based class definitions (`SolutionNet`, `ModulationNet`, `CoeffNet`, `EdgeNet`) for alternative implementations

### `code/util_learn_quad.py`
The core library. Key functions:
- `initialize_mlp_xavier` — Xavier-initialized MLP parameter tree
- `compute_J_zero_beta_value` — precomputes Jacobi polynomial boundary values needed for asymptotic expansions
- `family_root` — computes a single bulk quadrature node (root of the learned polynomial) and its weight via asymptotic expansion
- `family_edge_compute` — computes near-boundary (Lobatto-type) quadrature nodes and weights
- `weight_fn` — evaluates the learned weight function `w_θ(x)` at a point
- `unpack_params` — unpacks a flat parameter vector into a nested MLP parameter tree
- `plot_save_loss_dict` — saves a loss curve plot to disk

### `code/util.py`
Earlier utility functions; largely superseded by `util_learn_quad.py`.

### `code/PDE/*/main.py`
Each PDE experiment follows the same structure:
1. `parse_args` — experiment hyperparameters
2. `compute_param_num` — counts parameters in a network given its layer sizes
3. `init_distribution` — PDE-specific initial condition
4. `right_boundary` / `left_boundary` — boundary condition helpers
5. `compute_cfl` — CFL number utility
6. `compute_integral` — assembles the full learned quadrature rule for one training step
7. `l2_relative_error` / `gen_testdata` — loads reference data and computes test error
8. `main` — training loop

---

## Installation

```bash
git clone https://github.com/vsingh-group/learn-quad.git
cd learn-quad
pip install -r req.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `jax` / `jaxlib` | 0.4.18 | Automatic differentiation, JIT compilation |
| `optax` | 0.1.7 | Gradient-based optimizers (Adam) |
| `equinox` | 0.11.2 | Neural network building blocks |
| `numpy` | 1.24.3 | Array operations |
| `scipy` | 1.11.2 | Scientific computing utilities |
| `matplotlib` | 3.8.0 | Visualization |
| `pyDOE` | 0.3.8 | Latin hypercube sampling |

> The `req.txt` includes CUDA 12 variants of `jaxlib`. For CPU-only or different CUDA versions, install JAX separately following the [JAX installation guide](https://github.com/google/jax#installation).

---

## Running the Experiments

Each PDE experiment is run from its own directory. Reference test data (`.npy` or `.mat` files) must be placed in the same directory as `main.py` before running.

### Burgers Equation
```bash
cd code/PDE/burger
# Requires: Burgers.npz  (standard Burgers benchmark, e.g. from Raissi et al.)
python main.py --exp burger_run --NEpoch 100000 --lr 0.001 --penalty 10
```

### Allen-Cahn Equation
```bash
cd code/PDE/AC
# Requires: usol_D_0.001_k_5.mat
python main.py --exp ac_run --NEpoch 200000 --lr 0.0001 --penalty 1
```

### Convection Equation
```bash
cd code/PDE/convection
# Requires: convX.npy, convY.npy
python main.py --exp conv_run --NEpoch 200000 --lr 0.001 --penalty 10000 --dtype float32
```

### Diffusion Equation
```bash
cd code/PDE/diffusion
# Requires: test_x.npy, test_y.npy
python main.py --exp diff_run --NEpoch 100000 --lr 0.0001 --penalty 10
```

### Wave Equation
```bash
cd code/PDE/wave
# Requires: test_x.npy, test_y.npy
python main.py --exp wave_run --NEpoch 200000 --lr 0.001 --penalty 10000 --dtype float32
```

### 1D MHD Equations (Brio-Wu Riemann Problem)
```bash
cd code/PDE/mhd
# Requires: test_x.npy, test_y.npy
python Main.py --exp mhd_run --NEpoch 100000 --lr 0.001 --penalty 100
```

### Common Arguments

| Argument | Description | Default (burger) |
|----------|-------------|---------|
| `--seed` | Random seed | 52 |
| `--exp` | Experiment name (used for output directory) | varies |
| `--NEpoch` | Number of training epochs | 100000 |
| `--penalty` | Weight for IC/BC loss terms | varies |
| `--lr` | Learning rate | 0.001 |
| `--N_degree` | Maximum quadrature polynomial degree | 100 |
| `--min_degree` | Minimum quadrature polynomial degree | 70 |
| `--alpha` | Jacobi weight parameter α | 1 |
| `--beta` | Jacobi weight parameter β | 1 |
| `--family_width` | Width of the family network | 100 |
| `--family_depth` | Depth of the family network | 4 |
| `--modulation_width` | Width of the modulation sub-network | 10 |
| `--modulation_depth` | Depth of the modulation sub-network | 3 |
| `--solve_width` | Width of the solution network | 64 |
| `--solve_depth` | Depth of the solution network | 3–5 |
| `--print_freq` | Console print frequency (epochs) | 30 |
| `--plot_loss_freq` | Loss plot save frequency (epochs) | 1000 |
| `--model_save_freq` | Model checkpoint frequency (epochs) | 1000 |

### Output Structure

Each run creates:
```
Result/<exp_name>/Results/<n>/
    arguments.pkl       # Saved hyperparameters
    model/              # Model checkpoints and final weights
    quad/               # Quadrature visualizations
    sol/                # Solution plots
    train_sol/          # Training solution snapshots
    test_sol/           # Test solution snapshots
    data/               # Miscellaneous saved data
Result/<exp_name>/experiment.csv   # Hyperparameter log across all runs
```

---

## Implementing LearnQuad for a New PDE

To add a new PDE, copy any existing `main.py` and modify the following:

### 1. Define the initial and boundary conditions

```python
def init_distribution(x):
    # Return the true initial condition u(x, 0)
    # x is a scalar in [-1, 1]
    return jnp.sin(jnp.pi * x)   # example

def right_boundary(t, t2):
    return t2   # return the right boundary value (passed as t2)

def left_boundary(t, t1):
    return t1   # return the left boundary value (passed as t1)
```

### 2. Define the PDE residual inside `loss()`

The quadrature nodes `x_node` and time nodes `t_node` are already computed by `compute_integral`. Use JAX autodiff to compute the residual:

```python
# Example: u_t + c * u_x = 0  (convection)
residuals = jax.vmap(lambda x, t: (
    jax.grad(lambda t_: solution_net_forward_pass_v2(x, t_, sol_model))(t) +
    c * jax.grad(lambda x_: solution_net_forward_pass_v2(x_, t, sol_model))(x)
))(x_node, t_node_perm)
loss_pde = jnp.mean(residuals**2)
```

For vector-valued PDEs (like MHD), use `jax.jacobian` instead of `jax.grad`.

### 3. Prepare reference test data

Generate a reference solution using any accurate numerical method and save:
```python
# test_x.npy: shape (N, 2), each row is [x, t]
# test_y.npy: shape (N,) or (N, d) for d-component solutions
import numpy as np
np.save('test_x.npy', X_test)
np.save('test_y.npy', y_test)
```

### 4. Adjust network output size

The solution network output size should match the number of PDE unknowns:
```python
# Scalar PDE (Burgers, diffusion, wave, convection):
solution_model_size = [2] + [args.solve_width]*args.solve_depth + [1]

# Vector PDE (MHD with 7 conserved variables):
solution_model_size = [2] + [args.solve_width]*args.solve_depth + [7]
```

### 5. Tune the quadrature degree range

`--min_degree` and `--N_degree` control the range of polynomial degrees sampled during training. Higher degrees give more quadrature points. As a starting point:
- Simple smooth PDEs: `--min_degree 25 --N_degree 35`
- Moderate complexity: `--min_degree 70 --N_degree 100`
- High-frequency or discontinuous solutions: `--min_degree 970 --N_degree 1025`

---

## Citation

```bibtex
@inproceedings{learnquad2025,
  title     = {PINNs with Learnable Quadrature},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
}
```

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: Sourav Pal (spal9@wisc.edu)
