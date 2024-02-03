#include <stdio.h>
extern "C" __device__ void add_float_arrays_cuda_kernel_forward(float* a, float* b, float* c);


__global__
void my_kernel(float* a, float* b, float* c)
{
    add_float_arrays_cuda_kernel_forward(a, b, c);
}


int main(void)
{
    float a[4] = {1, 2, 3, 4};
    float b[4] = {1, 2, 3, 4};
    float c[4] = {0, 0, 0, 0};
    // my_kernel<<<1, 1>>>(a, b, c);
    // add_float_arrays_cuda_kernel_forward(c, a, b);
    printf("%f %f %f %f\n", c[0], c[1], c[2], c[3]);

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