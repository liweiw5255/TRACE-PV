set(CMAKE_CUDA_COMPILER "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.3.52")
set(CMAKE_CUDA_DEVICE_LINKER "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "12.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/targets/x86_64-linux/lib/stubs;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/include/c++/12.3.0;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/include/c++/12.3.0/x86_64-pc-linux-gnu;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/include/c++/12.3.0/backward;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib/gcc/x86_64-pc-linux-gnu/12.3.0/include-fixed;/usr/local/include;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/include;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/targets/x86_64-linux/lib/stubs;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/targets/x86_64-linux/lib;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib/gcc/x86_64-pc-linux-gnu/12.3.0;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib/gcc;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib64;/lib64;/usr/lib64;/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.2.0/gcc-12.3.0-gte3dhuw5dryttvxvjbbxdqxuz2panwt/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
