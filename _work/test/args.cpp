#include<Kokkos_Core.hpp>


int main(int argc, char** argv) {

  Kokkos::InitArguments args = {};
  Kokkos::InitArguments a1 = {};
  Kokkos::InitArguments a2 = {};
  Kokkos::InitArguments a3 = {};

  printf("num_threads = %d\n", args.num_threads);
  printf("num_numa = %d\n", args.num_numa);
  printf("device_id = %d\n", args.device_id);
  printf("ndevices = %d\n", args.ndevices);
  printf("skip_device = %d\n", args.skip_device);
  printf("disable_warnings = %d\n", args.disable_warnings);

  Kokkos::initialize(args);
  // add scoping to ensure my_view destructor is called before Kokkos::finalize  
  {
     Kokkos::View<double*> my_view("my_view", 10);
  }
 
  Kokkos::finalize();
  
}