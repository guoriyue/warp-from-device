#include <stdio.h>
#include "warp/warp/native/builtin.h"

extern "C" __global__ void add_float_arrays_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_dest,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> var_b);

__global__
void my_kernel(wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_dest,
    wp::array_t<wp::float32> var_a,
    wp::array_t<wp::float32> var_b)
{
    printf("var_dest.data[0]: %d\n", var_dest.data[0]);
    printf("var_dest.data[1]: %d\n", var_dest.data[1]);
    printf("var_dest.data[2]: %d\n", var_dest.data[2]);
    add_float_arrays_cuda_kernel_forward<<<1,1>>>(dim, var_dest, var_a, var_b);
    printf("var_dest.data[0]: %d\n", var_dest.data[0]);
    printf("var_dest.data[1]: %d\n", var_dest.data[1]);
    printf("var_dest.data[2]: %d\n", var_dest.data[2]);
}


int main(void)
{
    int num_items = 8;
    wp::launch_bounds_t bounds_cuda;
    bounds_cuda.ndim = 1;
    bounds_cuda.shape[0] = num_items;//??
    bounds_cuda.size = num_items;

    wp::array_t<float> var_dest, var_a, var_b;

    float a[8] = {1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8};
    float b[8] = {100.,200.,300.,400.,500.,600.,700.,800.};
    float c[8] = {-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.};

    float* cumem_a;
    float* cumem_b;
    float* cumem_c;
    cudaMalloc((void**)&cumem_a, sizeof(float) * 8);
    cudaMalloc((void**)&cumem_b, sizeof(float) * 8);
    cudaMalloc((void**)&cumem_c, sizeof(float) * 8);
    cudaMemcpy(cumem_a, a, sizeof(float) * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(cumem_b, b, sizeof(float) * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(cumem_c, c, sizeof(float) * 8, cudaMemcpyHostToDevice);

    var_dest.ndim = 1;
    var_dest.shape[0] = num_items;
    var_dest.strides[0] = sizeof(float);
    var_dest.data = cumem_c;

    var_a.ndim = 1;
    var_a.shape[0] = num_items;
    var_a.strides[0] = sizeof(float);
    var_a.data = cumem_a;

    var_b.ndim = 1;
    var_b.shape[0] = num_items;
    var_b.strides[0] = sizeof(float);
    var_b.data = cumem_b;

    my_kernel<<<1, 1>>>(bounds_cuda, var_dest, var_a, var_b);
    cudaMemcpy(c, var_dest.data, sizeof(float) * 8, cudaMemcpyDeviceToHost);
    printf("%f %f %f %f %f %f %f %f\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);

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