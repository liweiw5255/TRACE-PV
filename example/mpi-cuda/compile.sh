module load cuda/12.3.0 openmpi/4.1.6 gcc/12.3.0

nvcc -o multi_gpu multi_gpu.cu -I/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/openmpi-4.1.6-o24xaglwzn6ubwum4bavwp34qy7vowfs/include -L/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/openmpi-4.1.6-o24xaglwzn6ubwum4bavwp34qy7vowfs/lib -lmpi
