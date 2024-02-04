import numpy as np
import warp as wp

wp.init()
device = "cuda"


@wp.kernel
def add_float(dest: wp.array(dtype=wp.float32),
             a: wp.array(dtype=wp.float32),
             b: wp.array(dtype=wp.float32)):

    tid = wp.tid()
    dest[tid] = a[tid]+b[tid]



# def example_add_float(device, n):
#     dest = wp.zeros(n=n, dtype=wp.float32, device=device)
#     a = wp.array(np.linspace(0.5, 0.9, n), dtype=wp.float32, device=device)
#     b = wp.array(np.linspace(100, 110, n), dtype=wp.float32, device=device)
#     print("dir(a)=", dir(a))
#     print("a.ndim=",a.ndim)
#     print("a.shape=",a.shape)
#     print("a.strides=",a.strides)
   
#     wp.launch(add_float_arrays, dim=n, inputs=[dest, a, b], device=device)
#     print("dest.numpy()=",dest.numpy())
   
# example_add_float_arrays(device=device, n=8)


# Kernel cache: /home/eecs/zhen/.cache/warp/0.11.0
# dir(a)= ['__array_interface__', '__class__', '__class_getitem__', '__ctype__', '__cuda_array_interface__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__matmul__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_alloc_grad', '_array_interface', '_grad', '_init_annotation', '_init_from_data', '_init_from_ptr', '_init_new', '_is_protocol', '_requires_grad', '_vars', 'assign', 'capacity', 'contiguous', 'cptr', 'ctype', 'device', 'dtype', 'fill_', 'flatten', 'grad', 'is_contiguous', 'is_transposed', 'list', 'ndim', 'numpy', 'owner', 'pinned', 'ptr', 'requires_grad', 'reshape', 'shape', 'size', 'strides', 'to', 'transpose', 'vars', 'view', 'zero_']
# a.ndim= 1
# a.shape= (8,)
# a.strides= (4,)
# Module __main__ load on device 'cpu' took 1082.95 ms
# dest.numpy()= [100.5      101.98572  103.47143  104.95714  106.442856 107.92857
#  109.41428  110.9     ]




# Warp 0.11.0 initialized:
#    CUDA Toolkit: 11.5, Driver: 12.2
#    Devices:
#      "cpu"    | x86_64
#      "cuda:0" | NVIDIA RTX A6000 (sm_86)
#      "cuda:1" | NVIDIA RTX A6000 (sm_86)
#      "cuda:2" | NVIDIA RTX A6000 (sm_86)
#      "cuda:3" | NVIDIA RTX A6000 (sm_86)
#      "cuda:4" | NVIDIA RTX A6000 (sm_86)
#      "cuda:5" | NVIDIA RTX A6000 (sm_86)
#      "cuda:6" | NVIDIA RTX A6000 (sm_86)
#      "cuda:7" | NVIDIA RTX A6000 (sm_86)
#    Kernel cache: /home/eecs/zhen/.cache/warp/0.11.0
# dir(a)= ['__array_interface__', '__class__', '__class_getitem__', '__ctype__', '__cuda_array_interface__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__matmul__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_alloc_grad', '_array_interface', '_grad', '_init_annotation', '_init_from_data', '_init_from_ptr', '_init_new', '_is_protocol', '_requires_grad', '_vars', 'assign', 'capacity', 'contiguous', 'cptr', 'ctype', 'device', 'dtype', 'fill_', 'flatten', 'grad', 'is_contiguous', 'is_transposed', 'list', 'ndim', 'numpy', 'owner', 'pinned', 'ptr', 'requires_grad', 'reshape', 'shape', 'size', 'strides', 'to', 'transpose', 'vars', 'view', 'zero_']
# a.ndim= 1
# a.shape= (8,)
# a.strides= (4,)
# Module __main__ load on device 'cuda:0' took 401.98 ms
# dest.numpy()= [100.5      101.98572  103.47143  104.95714  106.442856 107.92857
#  109.41428  110.9     ]

