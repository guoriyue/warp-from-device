
#define WP_NO_CRT
#include "warp/warp/native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)


// block_max.py:26
static __device__ void cuda_block_max(
    wp::array_t<wp::float32> result,
    wp::array_t<wp::float32> a,
    wp::int32 tid)
{

    __shared__ int sdata[64];

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
    }


// block_max.py:26
static __device__ void adj_cuda_block_max(
    wp::array_t<wp::float32> result,
    wp::array_t<wp::float32> a,
    wp::int32 tid,
    wp::array_t<wp::float32> & adj_result,
    wp::array_t<wp::float32> & adj_a,
    wp::int32 & adj_tid)
{
}



extern "C" __global__ void block_max_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_result,
    wp::array_t<wp::float32> var_a)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        //---------
        // forward
        // def block_max(result: wp.array(dtype=wp.float32),                                      <L 31>
        // tid = wp.tid()                                                                         <L 33>
        var_0 = builtin_tid1d();
        // cuda_block_max(result, a, tid)                                                         <L 34>
        cuda_block_max(var_result, var_a, var_0);
    }
}

extern "C" __global__ void block_max_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_result,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> adj_result,
    wp::array_t<wp::float32> adj_a)
{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        //---------
        // primal vars
        wp::int32 var_0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        //---------
        // forward
        // def block_max(result: wp.array(dtype=wp.float32),                                      <L 31>
        // tid = wp.tid()                                                                         <L 33>
        var_0 = builtin_tid1d();
        // cuda_block_max(result, a, tid)                                                         <L 34>
        cuda_block_max(var_result, var_a, var_0);
        //---------
        // reverse
        adj_cuda_block_max(var_result, var_a, var_0, adj_result, adj_a, adj_0);
        // adj: cuda_block_max(result, a, tid)                                                    <L 34>
        // adj: tid = wp.tid()                                                                    <L 33>
        // adj: def block_max(result: wp.array(dtype=wp.float32),                                 <L 31>
        continue;
    }
}

