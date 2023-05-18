cudaPath=/usr/local/cuda_save_11.2
CC = 75
binDir=./bin

COMPILER_CPP=-x c++ --compiler-options -fpermissive
COMPILER_CUDA=-x cu

NVCC=nvcc
LINK=-lm
OPENMP=--compiler-options -fopenmp
INCLUDES=-I${cudaPath}/samples/common/inc
FLAGS = -DSM_${CC} -arch=sm_${CC} -lineinfo -Xcompiler=-O3 -Xptxas=-v
LINK=-lm

all: build

build: cuda

main.o: main.c
	$(NVCC) $(COMPILER_CPP) -DCORRECTNESS $(OPENMP) $(INCLUDES) $(FLAGS) -o $@ -c $<

mmio.o: mmio.c
	$(NVCC) $(COMPILER_CUDA) $(OPENMP) $(INCLUDES) -o $@ -c $<

parallel_conversions.o: ./parallel/parallel_conversions.c
	$(NVCC) $(COMPILER_CPP) $(OPENMP) $(INCLUDES) -o $@ -c $<

parallel_product.o: ./parallel/parallel_product.cu
	$(NVCC) $(COMPILER_CUDA) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

cuda: main.o mmio.o parallel_conversions.o parallel_product.o
	$(NVCC) $(OPENMP) $(LINK) $^ -o $@

adder_dcop_32: build
	./cuda ../Matrici/adder_dcop_32/adder_dcop_32.mtx 

Cube_Coup_dt0: build
	./cuda ../Matrici/Cube_Coup_dt0/Cube_Coup_dt0.mtx 

dc1: build
	./cuda ../Matrici/dc1/dc1.mtx 

PR02R: build
	./cuda ../Matrici/PR02R/PR02R.mtx 

raefsky2: build
	./cuda ../Matrici/raefsky2/raefsky2.mtx 

mhd4800a: build
	./cuda ../Matrici/mhd4800a/mhd4800a.mtx 

bcsstk17: build
	./cuda ../Matrici/bcsstk17/bcsstk17.mtx 

amazon0302: build
	./cuda ../Matrici/amazon0302/amazon0302.mtx 

ML_Laplace: build
	./cuda ../Matrici/ML_Laplace/ML_Laplace.mtx 

load: 
	module load gnu/ mpich/ cuda/ 