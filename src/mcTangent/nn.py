from typing import Tuple, NamedTuple, Optional, Union, Iterable, Any
import time
import os
import wandb
import shutil
import functools
import warnings
import numpy as np
import re
import jax
import jax.tree_util as jtr
import jax.numpy as jnp
import jax.random as jrand
import jax.image as jim
from jax import value_and_grad, vmap, jit, lax, pmap
import json
import pickle
import haiku as hk
import optax
import optax._src.base as opbase
import optax._src.utils as oputils
import optax._src.combine as opcomb
import optax._src.alias as opalias
import matplotlib.pyplot as plt
import chex

warnings.simplefilter('default', UserWarning)
"""
mcTangent NN generator
"""


def create(architecture: str = "DENSE", *a, **k):
    """
    Returns an mcTangent module based upon the input net architecture
    """
    if architecture.upper() not in DICT_ARCHITECTURES.keys():
        warnings.warn(message=f"\nInvalid architecture type {architecture.upper()} requested. Choose from:\n{[key for key in DICT_ARCHITECTURES.keys()]}.\nReturning default dense net",
                      category=UserWarning)
        return DICT_ARCHITECTURES["DENSE"]()
    else:
        try:
            return DICT_ARCHITECTURES[architecture.upper()](*a, **k)
        except:
            warnings.warn(
                "\nInvalid architecture arguments given, returning default network of specified type")
            return DICT_ARCHITECTURES[architecture.upper()]()


class mcT_net_dense(hk.Module):
    """
    Creates a dense linear mcTangent NN with the specified number of layers
    """

    def __init__(self, num_layers: int = 1, layer_sizes: Union[int, Iterable[int]] = 1, activations: Optional[Union[str, Iterable[str]]] = 'RELU',
                 out_size: Union[int, Iterable[int]] = 1, name: Optional[str] = None):
        super().__init__(name=name)

        self.num_layers = num_layers
        self.layer_sizes = [layer_sizes] * \
            num_layers if type(layer_sizes) == int else layer_sizes
        self.activations = [DICT_ACTIVATIONS[activations.upper()]]*num_layers if type(
            activations) == str else [DICT_ACTIVATIONS[fn.upper()] for fn in activations]
        assert self.num_layers == len(self.layer_sizes)
        assert self.num_layers == len(self.activations)

        self.out_size = out_size if type(
            out_size) == int else jnp.prod(out_size)
        self._create_net()

    def __call__(self, input):
        out = self.net(jnp.ravel(input))
        return jnp.reshape(out, (input.shape))

    def _create_net(self):
        sequence = []
        for size, activation in zip(self.layer_sizes, self.activations):
            if activation is not None:
                sequence += [hk.Linear(size), activation]
            else:
                sequence += [hk.Linear(size)]
        sequence.append(hk.Linear(self.out_size))
        self.net = hk.Sequential(sequence)

# TODO: CNN


# class mcT_net_cnn(hk.Module):
#     """
#     Creates a CNN with the specified number of layers
#     """

#     def __init__(
#         self,
#         ndims: int = 1,
#         layers: int = 1,
#         features: Union[int, Iterable[int]] = 1,
#         kernel_shape: Union[int, Tuple[int]] = 3,
#         stride: int = 1,
#         activations: Optional[Union[str, Iterable[str]]] = 'RELU',
#         out_size: int = 1,
#         name: Optional[str] = None
#     ):
#         super().__init__(name=name)

#         self.ndims = ndims
#         self.layers = layers

#         # check if features is compatible with layer count
#         if type(features) == int:
#             self.features = [features] * layers
#         elif len(features) != layers:
#             print("feature count and layer count do not match")
#             if len(features) > layers:
#                 self.features = features[:layers]
#             else:
#                 feature_list = features
#                 feature_list.append([1]*(layers-len(features)))
#                 self.features = feature_list
#         else:
#             self.features = features
        
#         # check if kernels are compatible with layers
#         if type(kernel_shape) == int:
#             self.kernel_shape = tuple([kernel_shape for dim in range(ndims)])
#         elif len(kernel_shape) != ndims:
#             print("kernel dims do not match network dims")
#             if len(kernel_shape) > ndims:
#                 self.kernel_shape = kernel_shape[:ndims]
#             else:
#                 kernel = list(kernel_shape)
#                 kernel.append([1]*(ndims-len(kernel)))
#                 self.kernel_shape = tuple(kernel)
#         else:
#             self.kernel_shape = kernel_shape
        
#         # check if activations are compatible
#         self.activations = activations
#         self.out_size = out_size


#     def __call__(self, state: jnp.ndarray) -> Any:


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
    "DENSE": lambda *args: hk.transform(lambda x: mcT_net_dense(*args)(x))
    # "CONVOLUTIONAL" : lambda *args: hk.transform(lambda x: mcT_net_cnn(*args)(x)),
}

DICT_ACTIVATIONS = {
    "RELU": jax.nn.relu,
    "TANH": jax.nn.hard_tanh,
    "ELU": jax.nn.elu,
    "NONE": None
}


class ScaleByEveState(NamedTuple):
    """State for the Eve algorithm."""
    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: opbase.Updates
    nu: opbase.Updates
    d: float
    f: float

# def __init__(self, lr_init: float = 1e-3, min_global: float = 0):
#     self.a1 = lr_init
#     self.b = [0.9,0.999,0.999]
#     self.c = 10
#     self.eps = 1e-8
#     self.t = 0
#     self.f = 1
#     self.f_star = min_global


def scale_by_eve(b1: float = 0.9,
                 b2: float = 0.999,
                 b3: float = 0.999,
                 c: float = 10.,
                 eps: float = 1e-8,
                 f_star: float = 0.,
                 mu_dtype: Optional[Any] = None,
                 ) -> opbase.GradientTransformation:
    """Rescale updates according to the Eve algorithm.

    References:
        [Hayashi et al, 2018](https://arxiv.org/abs/1611.01505)

    Args:
        b1: the exponential decay rate to track the first moment of past gradients.
        b2: the exponential decay rate to track the second moment of past gradients.
        b3: the exponential decay rate to track the sub-optimality.
        c: the clipping limit to prevent extreme global learning rate changes
        eps: a small constant applied to denominator outside of the square root
        (as in the Adam paper) to avoid dividing by zero when rescaling.
        f_star: estimation of the global minimum
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        An (init_fn, update_fn) tuple.
    """
    mu_dtype = oputils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        d = 1.
        f = 1.
        return ScaleByEveState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, d=d, f=f)

    def update_fn(updates: opbase.Updates, state: ScaleByEveState, f: float):
        mu = jtr.tree_map(lambda m, u: jnp.asarray(
            b1*m + (1-b1)*u), state.mu, updates)
        nu = jtr.tree_map(lambda v, u: jnp.asarray(
            b1*v + (1-b1)*u), state.nu, updates)
        count_inc = oputils.numerics.safe_int32_increment(state.count)
        mu_hat = jtr.tree_map(lambda m: jnp.asarray(m / (1-b1)), mu)
        nu_hat = jtr.tree_map(lambda v: jnp.asarray(v / (1-b2)), nu)
        if count_inc > 1:
            d_new = (jnp.abs(f - state.f)) / \
                (jnp.min(jnp.array([f, state.f])) - f_star)
            d_tilde = jnp.clip(d_new, 1/c, c)
            d = b3*state.d + (1-b3)*d_tilde
        else:
            d = 1.
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v) + eps) / d, mu_hat, nu_hat)
        mu = oputils.cast_tree(mu, mu_dtype)
        return updates, ScaleByEveState(count=count_inc, mu=mu, nu=nu, d=d, f=f)

    return opbase.GradientTransformation(init_fn, update_fn)


def eve(
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    b3: float = 0.999,
    c: float = 10.,
    eps: float = 1e-8,
    f_star: float = 0.,
    mu_dtype: Optional[Any] = None,
) -> opbase.GradientTransformation:
    """The Eve optimizer.

    Eve is an SGD variant with adaptive global and local learning rates. The `learning_rate`
    used for each weight is computed from estimates of first- and second-order
    moments of the gradients (using suitable exponential moving averages) as in ADAM.
    The global learning rate is scaled by some notion of sub-optimality and is increased
    when far from optimal and is decreased when approaching optimality

    References:
        Hayashi et al, 2018: https://arXiv.org/abs/1611.01505

    Args:
        learning_rate: this is the initial global scaling factor.
        b1: the exponential decay rate to track the first moment of past gradients.
        b2: the exponential decay rate to track the second moment of past gradients.
        b3: the exponential decay rate to track the sub-optimality.
        c: the clipping limit to prevent extreme global learning rate changes
        eps: a small constant applied to denominator outside of the square root
        (as in the Adam paper) to avoid dividing by zero when rescaling.
        f_star: estimation of the global minimum
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        the corresponding `GradientTransformation`.
    """
    return opcomb.chain(
        scale_by_eve(
            b1=b1, b2=b2, b3=b3, c=c, eps=eps, f_star=f_star, mu_dtype=mu_dtype),
        opalias._scale_by_learning_rate(learning_rate),
    )


if __name__ == "__main__":
    # dense architecture: dense, layers, width, activations, in, out
    net = create("dense", 1, 100, 'relu', 10, 10)
    print(net)
    testx = jnp.linspace(0, 1, 10)
    params = net.init(jrand.PRNGKey(1), testx)
    print(testx)
    print(net.apply(params, None, testx))
    print("\n-----Dense Pass-----\n")

    net = create('not implemented')
    print(net)

    optimizer = eve()
    print("\n-----Eve Creation Passed-----\n")
