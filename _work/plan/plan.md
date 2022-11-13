# Plan

## Statement

``` Plain
PennyLane's C++ simulator devices, `lightning.qubit`, `lightning.gpu` and `lightning.kokkos` provide different ways to accelerate quantum circuits and gradients.

`lightning.kokkos` is our Kokkos Core backed simulator, and supports running on all classical hardware that is supported by Kokkos https://kokkos.github.io/kokkos-core-wiki/. Multithreading support is performed at the gate level, and ensures that all gate-applications make the best use of the supporting hardware.

Since we support both C++ threads and OpenMP threads in the host execution space, we need explicit control of the number of executing threads for a given device to allow appropriate performance comparisons. The goal of this technical assessment is to extend `lightning.kokkos` to add Python-accessible controls to the Kokkos initialization routine.

To complete this task will require opening a pull-request (PR) against the https://github.com/PennyLaneAI/pennylane-lightning-kokkos/ repo and addressing the following points:

* Adding Python bindings to expose the `Kokkos::InitArguments` struct to the `lightning.kokkos` Python layer, which will be ignored by default. See here for more details https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Initialization.html#initialization-by-struct

* Ensure that the Kokkos initializer accepts the above struct at the C++ layer within the `StateVectorKokkos` class constructor. See https://github.com/PennyLaneAI/pennylane-lightning-kokkos/blob/main/pennylane_lightning_kokkos/src/simulator/StateVectorKokkos.hpp#L439 for where this is currently called.

* Add appropriate C++ and Python tests to verify the arguments are correctly validated.
 
* Evaluate a PennyLane code example of your choice, showcasing the effect different numbers of threads have on the overall runtime. You can do this for the `OPENMP` and `THREADS` backends, and add a plot to the PR to complete the demonstration.
```

## Stack

- Kokkos
- Pybinding
- Lightning
- Pennylane

## Install (done)

- Kokkos
- Pybinding
- Lightning
- Pennylane

Install with couple tweaks.

## Example

### Pick

[tutorial_backprop](https://pennylane.ai/qml/demos/tutorial_backprop.html)

### Run with lightning (OpenMP)

Can run.
OMP_NUM_THREADS seems to have no effect.
In StateVectorKokkos.hpp, I changed 

```
                printf("Hello initialize!\n");
                Kokkos::InitArguments args;
                // 8 (CPU) threads per NUMA region
                args.num_threads = 1;
                // 2 (CPU) NUMA regions per process
                args.num_numa = -1;
                // If Kokkos was built with CUDA enabled, use the GPU with device ID 1.
                // args.device_id = 1;
                Kokkos::initialize(args);
```

Setting `args.num_threads = 4;` slows down the code

```
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.01128472399977909 sec per loop
Gradient computation (best of 2): 2.1039355724988127 sec per loop
2.166667007957585
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.011237705999519676 sec per loop
Gradient computation (best of 2): 2.045110654000382 sec per loop
2.157639551907778
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.011380037998605985 sec per loop
Gradient computation (best of 2): 2.1474770059994626 sec per loop
2.184967295732349
```

compared with `args.num_threads = 1;`

```
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.008805997000308707 sec per loop
Gradient computation (best of 2): 1.5833353220004938 sec per loop
1.6907514240592718
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.008889769000234082 sec per loop
Gradient computation (best of 2): 1.6241683489988645 sec per loop
1.7068356480449438
(venv) vincentm@vincentm-ThinkPad-X1-Yoga-4th:~/repos/pennylane-lightning-kokkos/tests$ python test_gradient.py 
Hello initialize!
96
0.8879228929005824
Forward pass (best of 2): 0.009014065501105506 sec per loop
Gradient computation (best of 2): 1.5816104545010603 sec per loop
1.7307005762122571
```

So the problem is probably to small to benefit from OpenMP, but at least it has an effect.

### Perform scaling test

Pushed to later.

### Pass Python args to StateVectorKokkos

C++-bindings defined in `pennylane_lightning_kokkos/src/bindings/Bindings.cpp`
Kokkos init in `pennylane_lightning_kokkos/src/simulator/StateVectorKokkos.hpp`
Main class in `pennylane_lightning_kokkos/lightning_kokkos`

How is information passed through the binding?

### Pass C++ args to Kokkos

### Rerun scaling test & plot

### Redo with lightning (Threads)
