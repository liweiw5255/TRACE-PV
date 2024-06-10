cmake_minimum_required(VERSION 3.10)
project(MPI_MATLAB)

set(CMAKE_CXX_STANDARD 11)
set(MATLAB_ROOT "/software/commercial/matlab/R2023b")

# Adjust this path to the location of MatlabDataArray.hpp
include_directories(${MATLAB_ROOT}/extern/include)

# Set the C++ compiler to mpic++
set(CMAKE_CXX_COMPILER mpic++)

# Add the source files for your project
set(SOURCES
    main.cpp
    simulation.cpp
    clusterMerging.cpp
)

# Add the executable target
add_executable(MPI_MATLAB ${SOURCES})

# Link the MATLAB libraries
target_link_libraries(MPI_MATLAB
    -I${MATLAB_ROOT}/extern/include
    -L${MATLAB_ROOT}/extern/bin/glnxa64
    MatlabDataArray
    MatlabEngine
)

# Set the runtime library path
set_target_properties(MPI_MATLAB PROPERTIES
    BUILD_RPATH ${MATLAB_ROOT}/extern/bin/glnxa64
)