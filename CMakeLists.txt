# cmake -DCMAKE_BUILD_TYPE=Debug ..
cmake_minimum_required(VERSION 3.18)

# Set A2S_SIMULATION path
set(SIMULATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation") 
set(UTILITY_SIMULATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation/utility") 
set(A2S_SIMULATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation/a2s_simulation")
set(AVG_SIMULATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation/avg_simulation")
set(THERMAL_LOSS_SIMULATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation/thermal_loss_simulation")
set(PARAMETER_VARIATION_SRC "${CMAKE_SOURCE_DIR}/src/simulation/parameter_variation")

message("INFO is: ${A2S_SIMULATION_SRC}")

# Project name
project(TRACE_PV LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA toolkit path
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
# set(CUDA_TOOLKIT_ROOT_DIR /software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-11.8.0-in72fn46ydgmi5ak67tvzjll5dz4w43u)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_VERSION 12.3)
include_directories("${CUDA_INCLUDE_DIRS}")

# Set the debug flags for CUDA
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Add source files
set(CPP_SOURCES
    ${SIMULATION_SRC}/a2s_cuda_main.cpp
    ${UTILITY_SIMULATION_SRC}/a2s_cluster_merging.cpp
    ${UTILITY_SIMULATION_SRC}/functions.cpp
    ${PARAMETER_VARIATION_SRC}/parameter_variation.cpp

)

set(CUDA_SOURCES
    ${SIMULATION_SRC}/a2s_cuda_allocate.cu
    # CUDA KERNEL
    ${AVG_SIMULATION_SRC}/a2s_cuda_kernel_avg.cu
    ${A2S_SIMULATION_SRC}/a2s_cuda_kernel_a2s.cu
    ${A2S_SIMULATION_SRC}/a2s_cuda_kernel_calibration.cu
    ${THERMAL_LOSS_SIMULATION_SRC}/a2s_cuda_kernel_thermal.cu

    # CUDA DEVICE
    ${UTILITY_SIMULATION_SRC}/a2s_cuda_device.cu
    ${A2S_SIMULATION_SRC}/a2s_cuda_device_svm.cu
    ${A2S_SIMULATION_SRC}/a2s_cuda_device_ode.cu
    ${A2S_SIMULATION_SRC}/a2s_cuda_device_state_space.cu
)

# Add the executable target
add_executable(TRACE_PV ${CPP_SOURCES} ${CUDA_SOURCES})

# Set the output directory for the executable
set_target_properties(TRACE_PV PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Enable separable compilation
set_target_properties(TRACE_PV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Ensure that the --dc flag is used for device code compilation
set_source_files_properties(${CUDA_SOURCES} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA and MPI libraries
target_link_libraries(TRACE_PV ${CUDA_LIBRARIES} ${MPI_LIBRARIES})