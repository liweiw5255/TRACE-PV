# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/liweiw/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/liweiw/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liweiw/TRACE-PV_official/TRACE-PV

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liweiw/TRACE-PV_official/TRACE-PV/build

# Include any dependencies generated for this target.
include CMakeFiles/TRACE_PV.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/TRACE_PV.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/TRACE_PV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/TRACE_PV.dir/flags.make

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_main.cpp
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o.d -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_main.cpp

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.i"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_main.cpp > CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.i

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.s"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_main.cpp -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.s

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_functions.cpp
CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o.d -o CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_functions.cpp

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.i"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_functions.cpp > CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.i

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.s"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_functions.cpp -o CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.s

CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/parameter_variation/parameter_variation.cpp
CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o.d -o CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/parameter_variation/parameter_variation.cpp

CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.i"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/parameter_variation/parameter_variation.cpp > CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.i

CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.s"
	/software/openmpi/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/parameter_variation/parameter_variation.cpp -o CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.s

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_allocate.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_cuda_allocate.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu
CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu
CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_device.cu
CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/utility/a2s_cuda_device.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o: CMakeFiles/TRACE_PV.dir/flags.make
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o: /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu
CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o: CMakeFiles/TRACE_PV.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CUDA object CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o"
	/usr/local/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o -MF CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o.d -x cu -rdc=true -c /home/liweiw/TRACE-PV_official/TRACE-PV/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu -o CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target TRACE_PV
TRACE_PV_OBJECTS = \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o"

# External object files for target TRACE_PV
TRACE_PV_EXTERNAL_OBJECTS =

CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/build.make
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/deviceLinkLibs.rsp
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/deviceObjects1.rsp
CMakeFiles/TRACE_PV.dir/cmake_device_link.o: CMakeFiles/TRACE_PV.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CUDA device code CMakeFiles/TRACE_PV.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TRACE_PV.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TRACE_PV.dir/build: CMakeFiles/TRACE_PV.dir/cmake_device_link.o
.PHONY : CMakeFiles/TRACE_PV.dir/build

# Object files for target TRACE_PV
TRACE_PV_OBJECTS = \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o" \
"CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o"

# External object files for target TRACE_PV
TRACE_PV_EXTERNAL_OBJECTS =

/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_main.cpp.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_functions.cpp.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/parameter_variation/parameter_variation.cpp.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_cuda_allocate.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/avg_simulation/a2s_cuda_kernel_avg.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_a2s.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_kernel_calibration.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/thermal_loss_simulation/a2s_cuda_kernel_thermal.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/utility/a2s_cuda_device.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_svm.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_ode.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/src/simulation/a2s_simulation/a2s_cuda_device_state_space.cu.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/build.make
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/cmake_device_link.o
/home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV: CMakeFiles/TRACE_PV.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX executable /home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TRACE_PV.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/TRACE_PV.dir/build: /home/liweiw/TRACE-PV_official/TRACE-PV/bin/TRACE_PV
.PHONY : CMakeFiles/TRACE_PV.dir/build

CMakeFiles/TRACE_PV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/TRACE_PV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/TRACE_PV.dir/clean

CMakeFiles/TRACE_PV.dir/depend:
	cd /home/liweiw/TRACE-PV_official/TRACE-PV/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liweiw/TRACE-PV_official/TRACE-PV /home/liweiw/TRACE-PV_official/TRACE-PV /home/liweiw/TRACE-PV_official/TRACE-PV/build /home/liweiw/TRACE-PV_official/TRACE-PV/build /home/liweiw/TRACE-PV_official/TRACE-PV/build/CMakeFiles/TRACE_PV.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/TRACE_PV.dir/depend

