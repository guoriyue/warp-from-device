#include <stdio.h>
#include "/home/eecs/zhen/.cache/warp/0.11.0/bin/wp___main__.sm70.ptx"

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
    my_kernel<<<1, 1>>>(a, b, c);

    return 0;
}

// nvcc -o call.out --device-link wp___main__.fatbin /rscratch/zhendong/mfguo/warp/warp_cpp/call_ptx_from_device.cu --verbose