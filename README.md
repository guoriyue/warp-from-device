Call [NVIDIA Warp](https://github.com/nvidia/warp) kernels from device

Install Warp

```
pip install numpy
git clone https://github.com/NVIDIA/warp.git
cd warp
python build_lib.py --cuda_path=/usr/local/warp
pip install -e .
```

Run this Warp Python example to jit compile the example_add_float_array.py
```
python3 float_arrays_add.py
```

Note the Kernel cache path, it will contain the generated CUDA code.
Copy the CUDA file and save it in current directory.

```
nvcc call_wp_from_device.cu -rdc=true -lcudadevrt
./a.out
```


In call_wp_from_device.cu, we need to include:

```
#include "warp/warp/native/builtin.h"
#include "wp_float_arrays_add.cu"
```

To call the Warp kernel function, we need to convert our data into Warp variables.
```
extern "C" __global__ void add_float_arrays_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_dest,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> var_b)
```