from typing import Tuple, NamedTuple, Optional, Union, Iterable, Any
import time, os, wandb
import shutil, functools, warnings
import numpy as np
import re
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.image as jim
from jax import value_and_grad, vmap, jit, lax, pmap
import json
import pickle
import haiku as hk
import optax
import matplotlib.pyplot as plt

warnings.simplefilter('default', UserWarning)
"""
mcTangent NN generator
"""

def mcT_net(architecture: str = "DENSE", *a, **k):
    """
    Returns an mcTangent module based upon the input net architecture
    """
    if architecture.upper() not in DICT_ARCHITECTURES.keys():
        warnings.warn(message = f"\nInvalid architecture type {architecture.upper()} requested. Choose from:\n{[key for key in DICT_ARCHITECTURES.keys()]}.\nReturning default dense net",
                        category=UserWarning)
        return DICT_ARCHITECTURES["DENSE"]()
    else:
        try:
            return DICT_ARCHITECTURES[architecture.upper()](*a, **k)
        except:
            warnings.warn("\nInvalid architecture arguments given, returning default network of specified type")
            return DICT_ARCHITECTURES[architecture.upper()]()

class mcT_net_dense(hk.Module):
    """
    Creates a dense linear mcTangent NN with the specified number of layers
    """
    def __init__(self, num_layers: int = 1, layer_sizes: Union[int,Iterable[int]] = 1, activations: Optional[Union[str,Iterable[str]]] = 'RELU',
                    in_shape: Union[int,Iterable[int]] = 1, out_shape: Union[int,Iterable[int]] = 1, name: Optional[str] = None):
        super().__init__(name=name)

        self.num_layers = num_layers
        self.layer_sizes = [layer_sizes]*num_layers if type(layer_sizes) == int else layer_sizes
        self.activations = [DICT_ACTIVATIONS[activations.upper()]]*num_layers if type(activations) == str else [DICT_ACTIVATIONS[fn.upper()] for fn in activations]
        assert self.num_layers == len(self.layer_sizes)
        assert self.num_layers == len(self.activations)

        self.output_size = out_shape if type(out_shape) == int else jnp.prod(out_shape)
        self.out_shape = (out_shape,) if type(out_shape) == int else out_shape
        self._create_net()
    
    def __call__(self, input):
        return self.f(input)

    def _create_net(self):
        sequence = []
        for size, activation in zip(self.layer_sizes, self.activations):
            sequence += [hk.Linear(size), activation]
        sequence.append(hk.Linear(self.output_size))
        net = hk.Sequential(sequence)

        def f(x):
            out = net(jnp.ravel(x))
            return jnp.reshape(out, self.out_shape)
        
        self.f = f

# TODO: CNN
# class mcT_net_cnn(hk.Module):
#     """
#     Creates an mcTangent NN with the specified number of layers
#     """
#     def __init__(self, ndim: int = 1, num_layers: int = 1, layer_features: Union[int,Iterable[int]] = 1, kernel_shapes: Union[int,Iterable[Any]] = 1, output_shape: Union[int,Iterable[int]] = 1, name: Optional[str] = None):
#         super().__init__(name=name)

#         self.check_dims = functools.partial(self._check_dims,self,ndim)
#         self.check_shapes = functools.partial(self._check_shapes,self,ndim)

#         self.num_layers = num_layers

#         self.layer_features = [layer_features]*num_layers if type(layer_features) == int else layer_features
#         assert self.num_layers == len(self.layer_features)

#         # use single input as edge length of ndim cube kernel for all layers
#         if type(kernel_shapes) == int:
#             self.kernel_shapes = [[[[[kernel_shapes]*ndim]*features] for features in range(layer)] for layer in self.layer_features]
#         # use single iterable as shape for all layers and features, expanding if not all dims are given, truncating if too many
#         elif type(kernel_shapes) == Iterable[int]:
#             kernel_shape = self.check_dims(kernel_shapes)
#             self.kernel_shapes = [[[[kernel_shape] * features] for features in layer] for layer in num_layers]
#         # if iterable of shapes, check if sufficient count for num_layers
#         elif type(kernel_shapes) == Iterable[Iterable[int]]:
#             checked_shapes = []
#             for features in layer_features:
#                 checked_shapes = [self.check_shapes(len(features), self.check_dims(shape)) for features, shape
#                 kernel_shapes = [self.check_dims(shape) for shape in kernel_shapes]
#             self.kernel_shapes = kernel_shapes
#         # if iterable of iterable of shapes, check if each layer has sufficient kernels for each feature
#         else:
#             raise TypeError

#         self.output_size = output_shape if type (output_shape) == int else jnp.prod(output_shape)
#         self.output_shape = (output_shape,) if type (output_shape) == int else output_shape

#         self.f = self._create_net()
    
#     def __call__(self, inputs, *a, **k):
#         return self.f(inputs, *a, **k)
    
#     def _check_shapes(self, ndim, nshapes, shapes):
#         missing_shapes = nshapes - len(shapes)
#         shapes = [(shape.tolist()) for shape in shapes] if type(shapes) == jnp.ndarray else [(shape) for shape in shapes]
#         if missing_shapes > 0:
#             warnings.warn("Degenerate unit cube kernel used for last %s layers due to lack of input" %str(missing_shapes))
#             shapes += [[1]*ndim for _ in range(missing_shapes)]
#         elif missing_shapes < 0:
#             warnings.warn("Truncating excessive kernel shapes. %s too many were given" %str(-missing_shapes))
#             shapes = shapes[:missing_shapes]

#     def _check_dims(self, ndim, shape):
#         missing_dims = ndim - len(shape)
#         if missing_dims > 0:
#             pad_widths = (0,missing_dims)
#             shape = jnp.pad(jnp.array(shape),pad_widths,"constant",constant_values=1)
#         elif missing_dims < 0:
#             warnings.warn("Truncating excessive kernel dimensions. %s too many were given" %str(-missing_dims))
#             shape = shape[:missing_dims]
#         return shape
        
#     def _create_net(self) -> callable:
#         sequence = [hk.Flatten()]
#         for layer, size in range(self.num_layers), self.layer_sizes:
#             sequence += [hk.Linear(size), jax.nn.relu]
#         sequence.append(hk.Linear(self.output_size))
#         net = hk.Sequential(sequence)

#         def f(x):
#             out = net(x)
#             return jnp.reshape(out, self.output_shape)
        
#         return f



def mcT_fn(state: jnp.ndarray) -> jnp.ndarray:
    """Dense network with 1 layer of ReLU units"""
    n_fields = state.shape[0]
    n_faces = state.shape[1]
    mcT = hk.Sequential([
        hk.Flatten(),
        hk.Linear(5*n_faces*n_fields), jax.nn.relu,
        hk.Linear(n_faces)
    ])
    flux = mcT(state)
    flux = jnp.reshape(flux,(n_fields,n_faces,1,1))
    return flux

def save_params(params, path):
    params = jax.device_get(params)
    os.makedirs(path)
    with open(path, 'wb') as fp:
        pickle.dump(params, fp)

def load_params(path):
    assert os.path.exists(path), "Specified parameter save path does not exist"
    with open(path, 'rb') as fp:
        params = pickle.load(fp)
    return jax.device_put(params)


DICT_ARCHITECTURES = {
    "DENSE" : lambda *args: hk.transform(lambda x: mcT_net_dense(*args)(x))
    # "CONVOLUTIONAL" : lambda *args: hk.transform(lambda x: mcT_net_cnn(*args)(x)),
}

DICT_ACTIVATIONS = {
    "RELU" : jax.nn.relu,
    "TANH" : jax.nn.hard_tanh,
    "ELU" : jax.nn.elu
}

if __name__ == "__main__":
    # dense architecture: dense, layers, width, activations, in, out
    net = mcT_net("dense",1,100,'relu',10,10)
    print(net)
    testx = jnp.linspace(0,1,10)
    params = net.init(jrand.PRNGKey(1), testx)
    print(testx)
    print(net.apply(params, None, testx))
    print("\n-----Dense Pass-----\n")

    net = mcT_net('not implemented')
    print(net)