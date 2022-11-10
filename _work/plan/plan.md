# Plan

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

### Perform scaling test

### Pass Python args to StateVectorKokkos

C++-bindings defined in `pennylane_lightning_kokkos/src/bindings/Bindings.cpp`
Kokkos init in `pennylane_lightning_kokkos/src/simulator/StateVectorKokkos.hpp`
Main class in `pennylane_lightning_kokkos/lightning_kokkos`

How is information passed through the binding?

### Pass C++ args to Kokkos

### Rerun scaling test & plot

### Redo with lightning (Threads)
