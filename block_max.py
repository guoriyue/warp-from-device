import numpy as np
import warp as wp

wp.init()
device = "cuda"

snippet = """
    extern __shared__ int sdata[64];

    sdata[tid] = a[tid];
    __syncthreads();
    for (int s = 64 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    if(tid == 0) 
    {
        result[0] = sdata[0];
    }
    """

@wp.func_native(snippet)
def cuda_block_max(result: wp.array(dtype=wp.float32), a: wp.array(dtype=wp.float32), tid: int):
    ...

@wp.kernel
def block_max(result: wp.array(dtype=wp.float32),
             a: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    cuda_block_max(result, a, tid)

def example_block_max(device, n):
    result = wp.zeros(n=1, dtype=wp.float32, device=device)
    a = wp.array(np.array(np.linspace(100, 110, n)), dtype=wp.float32, device=device)
    wp.launch(block_max, dim=n, inputs=[result, a], device=device)
    print("result.numpy()=", result.numpy())
   
example_block_max(device=device, n=64)