#%%
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
# from pennylane_lightning_kokkos import kokkos_start, kokkos_end

engine = "lightning.kokkos"

# set the random seed
np.random.seed(42)

#%%
# # create a device to execute the circuit on
# dev = qml.device(engine, wires=3)

# @qml.qnode(dev, diff_method="parameter-shift")
# def circuit(params):
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.RZ(params[2], wires=2)

#     qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")

#     qml.RX(params[3], wires=0)
#     qml.RY(params[4], wires=1)
#     qml.RZ(params[5], wires=2)

#     qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern="ring")
#     return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

# # initial parameters
# params = np.random.random([6], requires_grad=True)

# print("Parameters:", params)
# print("Expectation value:", circuit(params))
# # fig, ax = qml.draw_mpl(circuit, decimals=2)(params)
# # plt.show()

# def parameter_shift_term(qnode, params, i):
#     shifted = params.copy()
#     shifted[i] += np.pi/2
#     forward = qnode(shifted)  # forward evaluation

#     shifted[i] -= np.pi
#     backward = qnode(shifted) # backward evaluation

#     return 0.5 * (forward - backward)

# # gradient with respect to the first parameter
# print(parameter_shift_term(circuit, params, 0))

# def parameter_shift(qnode, params):
#     gradients = np.zeros([len(params)])

#     for i in range(len(params)):
#         gradients[i] = parameter_shift_term(qnode, params, i)

#     return gradients

# print(parameter_shift(circuit, params))

# grad_function = qml.grad(circuit)
# print(grad_function(params)[0])
# print(qml.gradients.param_shift(circuit)(params))

#%%
n_wires = 8
n_layers = 4

dev = qml.device(engine, wires=n_wires)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.StronglyEntanglingLayers(params, wires=list(range(n_wires)))
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5) @ qml.PauliZ(6) @ qml.PauliZ(7))

# initialize circuit parameters
param_shape = qml.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=n_layers)
params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)
print(params.size)
print(circuit(params))
import timeit

reps = 2
num = 2
times = timeit.repeat("circuit(params)", globals=globals(), number=num, repeat=reps)
forward_time = min(times) / num

print(f"Forward pass (best of {reps}): {forward_time} sec per loop")

# create the gradient function
grad_fn = qml.grad(circuit)

times = timeit.repeat("grad_fn(params)", globals=globals(), number=num, repeat=reps)
backward_time = min(times) / num

print(f"Gradient computation (best of {reps}): {backward_time} sec per loop")
print(2 * forward_time * params.size)
# %%
