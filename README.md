# warp_cpp
Examples calling [NVIDIA Warp](https://github.com/nvidia/warp) precompiled (cached) kernels directly from C++ (without Python)

## Usage

Install Warp

```
pip install numpy
git clone https://github.com/NVIDIA/warp.git
cd warp
python build_lib.py --cuda_path=/usr/local/warp
pip install -e .
```
example_add_float_array.py has this Warp kernel:
```
@wp.kernel
def add_float_arrays(dest: wp.array(dtype=wp.float32),
             a: wp.array(dtype=wp.float32),
             b: wp.array(dtype=wp.float32)):

    tid = wp.tid()
    dest[tid] = a[tid]+b[tid]
```
Run this Warp Python example to jit compile the example_add_float_array.py
```
python example_add_float_array.py
Warp 0.8.2 initialized:
   CUDA Toolkit: 11.8, Driver: 12.1
   Devices:
     "cpu"    | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
     "cuda:0" | NVIDIA GeForce RTX 4090 Laptop GPU (sm_89)
   Kernel cache: C:\Users\erwin\AppData\Local\NVIDIA Corporation\warp\Cache\0.8.2
Module __main__ load on device 'cpu' took 15.43 ms
dest.numpy()= [100.5      101.98572  103.47143  104.95714  106.442856 107.92857
 109.41428  110.9     ]
 ```
Note the Kernel cache path, it will contain the compiled Warp kernel as CPU DLL or CUDA PTX binary.

Use cmake, compile and run the C++ example_add_float_array_cpu.cpp and example_add_float_array_cuda.cpp
Use the variable WARP_PATH to point to the location of the Warp source root
When running example_add_float_array_cpu, pass the location to the .dll / .so Warp compiled kernel file:
```
cmake -DWARP_PATH=/home/ecoumans/dev/warp_cpp/warp .
cmake --build .
./example_add_float_array_cpu /home/ecoumans/.cache/warp/0.8.2/bin/wp___main__.so
a:1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8
b:100 200 300 400 500 600 700 800
Sum:101.1 202.2 303.3 404.4 505.5 606.6 707.7 808.8
```

You can extract the Warp kernel C++ signature from the Warp generated c++ code in the cache/gen folder (gen\wp___main__.cpp)
```
// CPU entry points
void (*add_float_arrays_cpu_forward)(launch_bounds_t dim,
array_t<float32> var_dest,
array_t<float32> var_a,
array_t<float32> var_b);
```
The array_t and other Warp definitions are in the builtin.h header file.

Same for the cuda version, make sure to change device = "cpu" into device="cuda" to compile to CUDA/PTX.
Pass the path to the PTX file as first argument:
```
./example_add_float_array_cuda /home/ecoumans/.cache/warp/0.8.2/bin/wp___main__.sm70.ptx
hello cuda world
CUDA driver version:12010
CUDA device count:1
len=22741
Sum:101.1 202.2 303.3 404.4 505.5 606.6 707.7 808.8
```
