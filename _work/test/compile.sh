ROOT=$PWD/../../
g++ $1 \
-I${ROOT}/build/temp.linux-x86_64-3.8/_deps/kokkos-src/core/src \
-I${ROOT}/build/_deps/kokkos-build \
${ROOT}/build/_deps/kokkos-build/core/src/libkokkoscore.a \
-ldl \
-fopenmp \
-std=c++2a