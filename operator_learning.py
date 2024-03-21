# %%
import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu, elu
# from jax.config import config

import os
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import griddata
from sampling import add_border_to_map
from utils import SDF_RT, polygon_SDF


# %%
def interpolate_boundary_points(points, num_points):
    # Ensure the first and last points are the same to close the polygon
    if not onp.array_equal(points[0], points[-1]):
        points = onp.vstack([points, points[0]])

    # Calculate the lengths of each segment and their cumulative sum
    segment_lengths = onp.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = onp.cumsum(segment_lengths)
    total_length = cumulative_lengths[-1]

    if total_length == 0:
        raise ZeroDivisionError

    # Calculate the distance between each interpolated point
    distances_between_points = total_length / (num_points - 1)

    # Initialize the list of interpolated points and the current distance marker
    interpolated_points = [points[0]]
    current_distance = distances_between_points

    # Iterate through each segment to place points
    for i, segment_length in enumerate(segment_lengths):
        while current_distance <= cumulative_lengths[i]:
            # Calculate the ratio of the current distance within the current segment
            if i == 0:
                ratio = current_distance / segment_length
            else:
                ratio = (current_distance - cumulative_lengths[i - 1]) / segment_length

            # Linear interpolation to find the new point
            new_point = points[i] + ratio * (points[i + 1] - points[i])
            interpolated_points.append(new_point)

            # Update the current distance for the next point
            current_distance += distances_between_points

    # Ensure the correct number of points are generated by possibly adding the last point
    if len(interpolated_points) < num_points:
        interpolated_points.append(points[-1])

    return np.array(interpolated_points)

# %%
def normalize_points(points, origin, scale):
    points = (points - origin) / scale
    return points

# %%
def generate_bcs_training_data(points, m=50, P=50):

    # Input sample 
    u = points.reshape(1, -1)  # shape = (1, 2m)   (x1, y1, x2, y2, ...) 

    # Fixed sensors
    y = points  # shape = (m, 2)

    # Tile inputs
    u = np.tile(u, (P, 1))  # shape = (P, 2m) in this example P = m
    s = np.zeros((P, 1))

    return u, y, s 

# Geneate training data corresponding to PDE residual
def generate_res_training_data(key, points, m=50, P=500):

    # Input sample 
    u =  points.reshape(1, -1)  # shape = (1, 2m)   (x1, y1, x2, y2, ...) 

    y1, y2 = np.meshgrid(y1, y2)
    y = np.hstack((y1.flatten()[:, None], y2.flatten()[:, None])).astype(int)  # shape = (P, 2)

    origin = np.array([250, 250])
    y = (y - origin) / 250
    
    
    # Tile inputs
    u = np.tile(u, (P,1))  # shape = (P, 2m) 
    s = np.zeros((P, 1))

    return u, y, s 

# Geneate test data 
def generate_sdf_data(points, sdf_map, m=50, P=500):

    # Input sample 
    u =  points.reshape(1, -1)  # shape = (1, 2m)   (x1, y1, x2, y2, ...) 

    y1 = np.linspace(0, 500, P, endpoint=False)
    y2 = np.linspace(0, 500, P, endpoint=False)

    y1, y2 = np.meshgrid(y1, y2)
    y = np.hstack((y1.flatten()[:, None], y2.flatten()[:, None])).astype(int)  # shape = (P, 2)
    s = sdf_map[y[:, 0], y[:, 1]]
    origin = np.array([250, 250])
    y = (y - origin) / 250

    # Tile inputs
    u = np.tile(u, (P**2, 1))  # shape = (P, 2m) in this example P = m

    # Exact solution
    def signed_dist(y1, y2):
      return sdf_map[y1, y2]

    # s = signed_dist(y1, y2)
    
    return u, y, s 

# %%
class VisibilityDataset(data.Dataset):
    def __init__(self, data_path, num_samples, m, P, batch_size=64, generation_type='train_bcs', rng_key=random.PRNGKey(1234)):
        # Load data from data_path
        # This should include loading the images and corresponding labels

        self.data_path = data_path
        self.data_files = sorted(os.listdir(data_path))
        self.generation_type = generation_type
        self.m = m
        self.P = P
        self.key = rng_key
        self.batch_size = batch_size

        if self.generation_type == 'train_bcs' or 'train_res':
            self.valid_data_files = list(filter(lambda x: int(x.split('_')[-1].split('.')[0]) < num_samples, self.data_files))
        elif self.generation_type == 'test':
            self.valid_data_files = list(filter(lambda x: int(x.split('_')[-1].split('.')[0]) >= num_samples, self.data_files))
        else:
            raise ValueError("Give a valid value of generation_type")

        self.N = len(self.valid_data_files)
  
    def __len__(self):
        return self.N


    def __getitem__(self, idx):
        valid_data_files = onp.array(self.valid_data_files)
        file_paths = onp.random.choice(valid_data_files, (self.batch_size,), replace=False)
        
        u_list, y_list, s_list = [], [], []

        for file in file_paths:
          x = np.load(f'{self.data_path}/{file}')
          points = x['visibility_region_free']

          if self.generation_type == 'train_bcs':
              u, y, s = generate_bcs_training_data(points, self.m, self.P)
          elif self.generation_type == 'train_res':
              self.key, subkey = random.split(self.key)
              u, y, s = generate_res_training_data(subkey, points, self.m, self.P)
          elif self.generation_type == 'test':
              sdf_map = x['signed_distance_free']
              u, y, s = generate_sdf_data(points, sdf_map, self.m, self.P)
          u_list.append(u)
          y_list.append(y)
          s_list.append(s)

        u = np.stack(u_list).reshape(self.batch_size * self.P, -1)
        y = np.stack(y_list).reshape(self.batch_size * self.P, -1)
        s = np.stack(s_list).reshape(self.batch_size * self.P, -1)

        idx = random.choice(self.key, self.batch_size * self.P, (self.batch_size  * self.P,), replace=False)
        u = u[idx,:]
        y = y[idx,:]
        s = s[idx,:]

        inputs = (u, y)
        outputs = s
        return inputs, outputs


# %%
# Define the neural net
def MLP(layers, activation=relu):
  ''' Vanilla MLP'''
  def init(rng_key):
      def init_layer(key, d_in, d_out):
          k1, k2 = random.split(key)
          glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
          W = glorot_stddev * random.normal(k1, (d_in, d_out))
          b = np.zeros(d_out)
          return W, b
      key, *keys = random.split(rng_key, len(layers))
      params = list(map(init_layer, keys, layers[:-1], layers[1:]))
      return params
  def apply(params, inputs):
      for W, b in params[:-1]:
          outputs = np.dot(inputs, W) + b
          inputs = activation(outputs)
      W, b = params[-1]
      outputs = np.dot(inputs, W) + b
      return outputs
  return init, apply

# %%
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

# %%
# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):    
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=relu)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=relu)

        # Initialize
        branch_params = self.branch_init(rng_key = random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key = random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=1000, 
                                                                      decay_rate=0.9))
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()

        # Logger
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    def operator_net(self, params, u, y1, y2):
        y = np.stack([y1, y2])
        branch_params, trunk_params = params
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    # Define ODE/PDE residual
    def residual_net(self, params, u, y1, y2):
        s_y1 = grad(self.operator_net, argnums = 2)(params, u, y1, y2)
        s_y2 = grad(self.operator_net, argnums = 3)(params, u, y1, y2)
        res = s_y1**2 + s_y2**2
        return res

    # Define boundary loss
    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        # Compute loss
        loss = np.mean((outputs.flatten() - pred.flatten())**2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:, 1])
        # Compute loss
        loss = np.mean((outputs.flatten() - pred.flatten())**2)
        return loss    
    
    # Define total loss
    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = 100 * loss_bcs + loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, val_dataset, nIter = 10000):
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)
        val_data = iter(val_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            try:
              bcs_batch = next(bcs_data)
              res_batch = next(res_data)
              self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)
              if (it + 1) % 1 == 0:
                  val_batch = next(val_data)
                  self.opt_state = self.step(next(self.itercount), self.opt_state, val_batch, res_batch)
            except ZeroDivisionError:
                continue
            
            if it % 1 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_bcs' : loss_bcs_value, 
                                  'loss_res': loss_res_value})
       
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred

# %%
class BCSDataGenerator(data.Dataset):
    def __init__(self, batch_size, m, map_size, alpha, r) -> None:
        self.m, self.P_train = m, m
        self.batch_size = batch_size
        self.map_size = map_size
        self.origin = onp.array([map_size / 2, map_size / 2])
        self.scale = map_size / 2
        empty_map = onp.zeros((map_size, map_size), onp.uint8)
        self.empty_map = add_border_to_map(empty_map)
        self.alpha = alpha
        self.r = r
    
    def __getitem__(self, index):
        'Generates data containing batch_size samples'
        u, y, s = self.__data_generation()
        idx = onp.random.choice(u.shape[0], (self.batch_size,), replace=False)
        s = s[idx,:]
        y = y[idx,:]
        u = u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

    def __data_generation(self):
        'Generate one batch of data'
        N_train = self.batch_size // self.P_train + 1

        u_train, y_train, s_train = [], [], []
        for idx in range(N_train):
            pos = onp.random.uniform(low=0, high=self.map_size, size=(2,1)).astype(int)
            ort = onp.random.uniform(low=0, high= 2 * onp.pi, size=(1,1)).astype(int)
            position = onp.vstack([pos, ort]).squeeze()

            visibility_m = SDF_RT(position, self.alpha, self.r, 50, self.empty_map).astype(int)

            visibility_m = interpolate_boundary_points(visibility_m, self.m)
            u = normalize_points(visibility_m, self.origin, self.scale)
            u = u.reshape((-1, 1))

            u_train.append(onp.tile(u, (self.P_train, 1)))
            y_train.append(u)
            s_train.append(onp.zeros((self.P_train, 1)))
        
        u_train = np.stack(u_train)
        y_train = np.stack(y_train)
        s_train = np.stack(s_train)

        u_train = u_train.reshape(N_train * self.P_train,-1)
        y_train = y_train.reshape(N_train * self.P_train,-1)
        s_train = s_train.reshape(N_train * self.P_train,-1)

        return u_train, y_train, s_train

# %%
class ResDataGenerator(data.Dataset):
    def __init__(self, batch_size, m, P_r_train, map_size, alpha, r) -> None:
        self.m = m
        self.P_r_train = P_r_train
        self.batch_size = batch_size
        self.map_size = map_size
        self.origin = onp.array([map_size / 2, map_size / 2])
        self.scale = map_size / 2
        empty_map = onp.zeros((map_size, map_size), onp.uint8)
        self.empty_map = add_border_to_map(empty_map)
        self.alpha = alpha
        self.r = r
    
    def __getitem__(self, index):
        'Generates data containing batch_size samples'
        u, y, s = self.__data_generation()
        idx = onp.random.choice(u.shape[0], (self.batch_size,), replace=False)
        s = s[idx,:]
        y = y[idx,:]
        u = u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

    def __data_generation(self):
        'Generate one batch of data'
        N_train = self.batch_size // (self.P_r_train ** 2) + 1

        u_r_train, y_r_train, s_r_train = [], [], []
        for idx in range(N_train):
            pos = onp.random.uniform(low=0, high=self.map_size, size=(2,1)).astype(int)
            ort = onp.random.uniform(low=0, high= 2 * onp.pi, size=(1,1)).astype(int)
            position = onp.vstack([pos, ort]).squeeze()

            visibility_m = SDF_RT(position, self.alpha, self.r, 50, self.empty_map).astype(int)

            visibility_m = interpolate_boundary_points(visibility_m, self.m)
            u = normalize_points(visibility_m, self.origin, self.scale)
            u = u.reshape((-1, 1))

            y1 = onp.linspace(0, self.map_size, self.P_r_train, endpoint=False)
            y2 = onp.linspace(0, self.map_size, self.P_r_train, endpoint=False)

            y1, y2 = onp.meshgrid(y1, y2)
            y = onp.hstack((y1.flatten()[:, None], y2.flatten()[:, None])) # shape = (P, 2)
            y = normalize_points(y, self.origin, self.scale)

            u_r_train.append(onp.tile(u, (self.P_r_train**2,1)))
            y_r_train.append(y)
            s_r_train.append(onp.ones((self.P_r_train**2, 1)))
        
        u_r_train = np.stack(u_r_train)
        y_r_train = np.stack(y_r_train)
        s_r_train = np.stack(s_r_train)

        u_r_train = u_r_train.reshape(N_train * self.P_r_train**2,-1)
        y_r_train = y_r_train.reshape(N_train * self.P_r_train**2,-1)
        s_r_train = s_r_train.reshape(N_train * self.P_r_train**2,-1)

        return u_r_train, y_r_train, s_r_train

# %%
class ValDataGenerator(data.Dataset):
    def __init__(self, batch_size, m, P_r_train, map_size, alpha, r) -> None:
        self.m = m
        self.P_r_train = P_r_train
        self.batch_size = batch_size
        self.map_size = map_size
        self.origin = onp.array([map_size / 2, map_size / 2])
        self.scale = map_size / 2
        empty_map = onp.zeros((map_size, map_size), onp.uint8)
        self.empty_map = add_border_to_map(empty_map)
        self.alpha = alpha
        self.r = r
    
    def __getitem__(self, index):
        'Generates data containing batch_size samples'
        u, y, s = self.__data_generation()
        idx = onp.random.choice(u.shape[0], (self.batch_size,), replace=False)
        s = s[idx,:]
        y = y[idx,:]
        u = u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs

    def __data_generation(self):
        'Generate one batch of data'
        N_train = self.batch_size // (self.P_r_train ** 2) + 1

        u_v_train, y_v_train, s_v_train = [], [], []
        for idx in range(N_train):
            pos = onp.random.uniform(low=0, high=self.map_size, size=(2,1)).astype(int)
            ort = onp.random.uniform(low=0, high= 2 * onp.pi, size=(1,1)).astype(int)
            position = onp.vstack([pos, ort]).squeeze()

            visibility_m = SDF_RT(position, self.alpha, self.r, 50, self.empty_map).astype(int)
            y1 = onp.linspace(0, self.map_size, self.P_r_train, endpoint=False)
            y2 = onp.linspace(0, self.map_size, self.P_r_train, endpoint=False)

            y1, y2 = onp.meshgrid(y1, y2)
            y = onp.hstack((y1.flatten()[:, None], y2.flatten()[:, None])) # shape = (P, 2)
            
            sdf_map = onp.zeros_like(self.empty_map)
            for i, j in itertools.product(range(self.empty_map.shape[0]),range(self.empty_map.shape[1])):
                target_m = onp.array([i, j, 0])
                signed_distance_free = polygon_SDF(visibility_m, target_m[0:2])
                sdf_map[j, i] = signed_distance_free
              
            sdf_map = sdf_map / (self.scale * onp.sqrt(2))

            visibility_m = interpolate_boundary_points(visibility_m, self.m)
            u = normalize_points(visibility_m, self.origin, self.scale)
            u = u.reshape((-1, 1))

            y = normalize_points(y, self.origin, self.scale)
            s = sdf_map[y[:, 0].astype(int), y[:, 1].astype(int)]

            u_v_train.append(onp.tile(u, (self.P_r_train**2,1)))
            y_v_train.append(y)
            s_v_train.append(s)
        
        u_v_train = np.stack(u_v_train)
        y_v_train = np.stack(y_v_train)
        s_v_train = np.stack(s_v_train)

        u_v_train = u_v_train.reshape(N_train * self.P_r_train**2,-1)
        y_v_train = y_v_train.reshape(N_train * self.P_r_train**2,-1)
        s_v_train = s_v_train.reshape(N_train * self.P_r_train**2,-1)

        return u_v_train, y_v_train, s_v_train

# %%
m = 40  # number of input sensors
P_train = m   # number of output sensors
P_r_train = 100   # number of output sensors
map_size = 200
scale = map_size / 2
alpha = onp.pi / 3
r = 50

# %%
# Initialize model
branch_layers = [2 * m, 100, 100, 100, 100, 100]
trunk_layers =  [2, 100, 100, 100, 100, 100]
model = PI_DeepONet(branch_layers, trunk_layers)

# %%
with open('params_free_100.pkl', 'rb') as inp:
  params = pickle.load(inp)

model.opt_state = model.opt_init(params)

# %%
# Create data set
batch_size = 10000
bcs_dataset = BCSDataGenerator(batch_size=batch_size, m=m, map_size=map_size, alpha=alpha, r=r)
res_dataset = ResDataGenerator(batch_size=batch_size, m=m, P_r_train=P_r_train, map_size=map_size, alpha=alpha, r=r)
val_dataset = ValDataGenerator(batch_size=batch_size, m=m, P_r_train=P_r_train, map_size=map_size, alpha=alpha, r=r)

# %%
try:
  model.train(bcs_dataset, res_dataset, val_dataset, nIter=40000)
except KeyboardInterrupt:
  pass
finally:
  params = model.get_params(model.opt_state)
  with open('params_free_100_all.pkl', 'wb') as f:
      pickle.dump(params, f)

# %%
#Plot for loss function
plt.figure(figsize = (6,5))
# plt.plot(model.loss_log, lw=2)
plt.plot(model.loss_bcs_log, lw=2, label='bcs')
plt.plot(model.loss_res_log, lw=2, label='res')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
