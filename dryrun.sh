_NVVM_BRANCH_=nvvm
_SPACE_= 
_CUDART_=cudart
_HERE_=/usr/local/cuda/bin
_THERE_=/usr/local/cuda/bin
_TARGET_SIZE_=
_TARGET_DIR_=
_TARGET_DIR_=targets/x86_64-linux
TOP=/usr/local/cuda/bin/..
NVVMIR_LIBRARY_DIR=/usr/local/cuda/bin/../nvvm/libdevice
LD_LIBRARY_PATH=/usr/local/cuda/bin/../lib:/usr/local/cuda/bin/../lib:/usr/local/cuda/bin/../lib:/home/eecs/zhen/custom_libs:/home/eecs/zhen/.mujoco/mjpro150/bin/:/rscratch/zhendong/yang_tasc/openmpi_install/lib
PATH=/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/usr/local/cuda/bin/../nvvm/bin:/usr/local/cuda/bin:/home/eecs/zhen/.cargo/bin:/home/eecs/zhen/google-cloud-sdk/bin:/rscratch/zhendong/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda/bin:/rscratch/zhendong/yang_tasc/openmpi_install/bin
INCLUDES="-I/usr/local/cuda/bin/../targets/x86_64-linux/include"  
LIBRARIES=  "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"
CUDAFE_FLAGS=
PTXAS_FLAGS=
gcc -D__CUDA_ARCH_LIST__=520 -E -x c++ -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=128 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "call_wp_max_block.cu" -o "/tmp/tmpxft_002706ed_00000000-5_call_wp_max_block.cpp4.ii" 
cudafe++ --c++14 --gnu_version=90400 --display_error_number --orig_src_file_name "call_wp_max_block.cu" --orig_src_path_name "/rscratch/zhendong/mfguo/warp/warp_cpp/call_wp_max_block.cu" --allow_managed  --device-c  --m64 --parse_templates --gen_c_file_name "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.cpp" --stub_file_name "tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.stub.c" --gen_module_id_file --module_id_file_name "/tmp/tmpxft_002706ed_00000000-4_call_wp_max_block.module_id" "/tmp/tmpxft_002706ed_00000000-5_call_wp_max_block.cpp4.ii" 
gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -E -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=128 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include "cuda_runtime.h" -m64 "call_wp_max_block.cu" -o "/tmp/tmpxft_002706ed_00000000-9_call_wp_max_block.cpp1.ii" 
cicc --c++14 --gnu_version=90400 --display_error_number --orig_src_file_name "call_wp_max_block.cu" --orig_src_path_name "/rscratch/zhendong/mfguo/warp/warp_cpp/call_wp_max_block.cu" --allow_managed  --device-c   -arch compute_52 -m64 --no-version-ident -ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "tmpxft_002706ed_00000000-3_call_wp_max_block.fatbin.c" -tused --module_id_file_name "/tmp/tmpxft_002706ed_00000000-4_call_wp_max_block.module_id" --gen_c_file_name "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.c" --stub_file_name "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.stub.c" --gen_device_file_name "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.gpu"  "/tmp/tmpxft_002706ed_00000000-9_call_wp_max_block.cpp1.ii" -o "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.ptx"
ptxas -arch=sm_52 -m64 --compile-only  "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.ptx"  -o "/tmp/tmpxft_002706ed_00000000-10_call_wp_max_block.sm_52.cubin" 
fatbinary -64 --cmdline="--compile-only  " --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " "--image3=kind=elf,sm=52,file=/tmp/tmpxft_002706ed_00000000-10_call_wp_max_block.sm_52.cubin" "--image3=kind=ptx,sm=52,file=/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.ptx" --embedded-fatbin="/tmp/tmpxft_002706ed_00000000-3_call_wp_max_block.fatbin.c"  --device-c
rm /tmp/tmpxft_002706ed_00000000-3_call_wp_max_block.fatbin
gcc -D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -c -x c++  -DCUDA_DOUBLE_MATH_FUNCTIONS "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"   -m64 "/tmp/tmpxft_002706ed_00000000-6_call_wp_max_block.cudafe1.cpp" -o "/tmp/tmpxft_002706ed_00000000-11_call_wp_max_block.o" 
nvlink -m64 --arch=sm_52 --register-link-binaries="/tmp/tmpxft_002706ed_00000000-7_a_dlink.reg.c"  -lcudadevrt   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -cpu-arch=X86_64 "/tmp/tmpxft_002706ed_00000000-11_call_wp_max_block.o"  -lcudadevrt  -o "/tmp/tmpxft_002706ed_00000000-12_a_dlink.sm_52.cubin" --host-ccbin "gcc"
fatbinary -64 --cmdline="--compile-only  " --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " -link "--image3=kind=elf,sm=52,file=/tmp/tmpxft_002706ed_00000000-12_a_dlink.sm_52.cubin" --embedded-fatbin="/tmp/tmpxft_002706ed_00000000-8_a_dlink.fatbin.c" 
rm /tmp/tmpxft_002706ed_00000000-8_a_dlink.fatbin
gcc -D__CUDA_ARCH_LIST__=520 -c -x c++ -DFATBINFILE="\"/tmp/tmpxft_002706ed_00000000-8_a_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"/tmp/tmpxft_002706ed_00000000-7_a_dlink.reg.c\"" -I. -D__NV_EXTRA_INITIALIZATION= -D__NV_EXTRA_FINALIZATION= -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  "-I/usr/local/cuda/bin/../targets/x86_64-linux/include"    -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=2 -D__CUDACC_VER_BUILD__=128 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=2 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -m64 "/usr/local/cuda/bin/crt/link.stub" -o "/tmp/tmpxft_002706ed_00000000-13_a_dlink.o" 
g++ -D__CUDA_ARCH_LIST__=520 -m64 -Wl,--start-group "/tmp/tmpxft_002706ed_00000000-13_a_dlink.o" "/tmp/tmpxft_002706ed_00000000-11_call_wp_max_block.o" -lcudadevrt   "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib"  -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group -o "a.out"