#include <stdio.h>
#include "warp/warp/native/builtin.h"
// #include "wp_block_max.cu"

extern "C" __global__ void block_max_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_result,
    wp::array_t<wp::float32> var_a);

__global__
void my_kernel(float* dest, float* a)
{
    int num_items = 64;
    wp::launch_bounds_t bounds_cuda;
    bounds_cuda.ndim = 1;
    bounds_cuda.shape[0] = num_items;
    bounds_cuda.size = num_items;

    wp::array_t<float> var_dest, var_a;

    var_dest.ndim = 1;
    var_dest.shape[0] = num_items;
    var_dest.strides[0] = sizeof(float);
    var_dest.data = dest;

    var_a.ndim = 1;
    var_a.shape[0] = num_items;
    var_a.strides[0] = sizeof(float);
    var_a.data = a;

    block_max_cuda_kernel_forward<<<1,64>>>(bounds_cuda, var_dest, var_a);
}

int main(void)
{
    int array_size = 64;
    float a[array_size];
    float dest[array_size];

    for (int i = 0; i < array_size; i++)
    {
        a[i] = 1.1 + i;
        dest[i] = -1.;
    }

    float* cumem_a;
    float* cumem_dest;
    cudaMalloc((void**)&cumem_a, sizeof(float) * array_size);
    cudaMalloc((void**)&cumem_dest, sizeof(float) * array_size);
    cudaMemcpy(cumem_a, a, sizeof(float) * array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cumem_dest, dest, sizeof(float) * array_size, cudaMemcpyHostToDevice);

    my_kernel<<<1, 1>>>(cumem_dest, cumem_a);
    cudaMemcpy(dest, cumem_dest, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    printf("%f\n", dest[0]);
    return 0;
}
// nvcc -arch=sm_70 -Xptxas="-v" -dlink -o memset.out /rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu

// nvcc -o call.out --device-link wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu --verbose

// nvcc -arch=sm_70 -Xptxas="-v" -dlink -o cptx /rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu

// nvcc -arch=sm_70 -o call_ptx_from_device.o -c /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu

// nvcc -arch=sm_70 -Xptxas="-v" -dlink /rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu


// nvcc -arch=sm_70 -Xptxas="-v" -dlink -o cptx.out /rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.fatbin call_ptx_from_device.o

// nvcc -arch=sm_70 -Xptxas="-v" -dlink -o cptx wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu

// You should be able to run ptx code directly from the cuda driver api with cuModuleLoadDataEx. There is an example here at page 5

// fatbinary --create="/rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.fatbin" --image3=kind=ptx,file=wp___main__.sm70.ptx,sm=70


// nvcc -arch=sm_70 -Xptxas="-v" -dlink -o cptx wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu
// ptxas --gpu-name sm_70 --verbose --output-file wp_ptxas_compile.ptx /rscratch/zhendong/mfguo/warp/warp_cpp/wp___main__.sm70.ptx

// gcc "call_ptx_from_device.cu" -o "call_ptx_from_device.cpp1.ii" 

// nvcc -dryrun call_ptx_from_device.cu  -rdc=true -lcudadevrt
// nvcc call_ptx_from_device.cu  -rdc=true -lcudadevrt