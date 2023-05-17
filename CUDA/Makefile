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

main.o: main.cu
	$(NVCC) $(COMPILER_CUDA) $(OPENMP) $(INCLUDES) $(FLAGS) -o $@ -c $<

mmio.o: mmio.c
	$(NVCC) $(COMPILER_CUDA) $(OPENMP) $(INCLUDES) -o $@ -c $<

parallel_conversions.o: ./parallel/parallel_conversions.c
	$(NVCC) $(COMPILER_CPP) $(OPENMP) $(INCLUDES) -o $@ -c $<

parallel_product.o: ./parallel/parallel_product.cu
	$(NVCC) $(COMPILER_CUDA) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

cuda: main.o mmio.o parallel_conversions.o parallel_product.o
	$(NVCC) $(OPENMP) $(LINK) $^ -o $@

run: build
	./cuda ../Matrici/adder_dcop_32/adder_dcop_32.mtx 

load: 
	module load gnu/ mpich/ cuda/ 