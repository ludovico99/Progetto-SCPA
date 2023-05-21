cudaPath=/usr/local/cuda_save_11.2
CC = 75
binDir=./bin

COMPILER_CPP=-x c++ --compiler-options -fpermissive
COMPILER_CUDA=-x cu

NVCC=nvcc
LINK=-lm
DEFINES= -DCORRECTNESS -DCUDA -DCSR
OPENMP=--compiler-options -fopenmp
INCLUDES=-I${cudaPath}/samples/common/inc
FLAGS = -DSM_${CC} -arch=sm_${CC} -lineinfo -Xcompiler=-O3 -Xptxas=-v
LINK=-lm

all: build

build: app

main.o: main.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS) -o $@ -c $<

mmio.o: mmio.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) -o $@ -c $<

conversions_parallel.o: conversions_parallel.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) -o $@ -c $<

conversions_serial.o: conversions_serial.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) -o $@ -c $<

parallel_product.o: ./CUDA/parallel_product.cu
	$(NVCC) $(COMPILER_CUDA) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

serial_product.o: serial_product.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

utils.o: utils.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

checks.o: checks.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

create_mtx_coo.o: create_mtx_coo.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

app: main.o mmio.o conversions_parallel.o parallel_product.o serial_product.o utils.o create_mtx_coo.o
	$(NVCC) $(OPENMP) $(DEFINES) $(LINK) $^ -o $@

copy-openMP:
	scp -i /home/ludovico99/.ssh/id_rsa -r /home/ludovico99/Scrivania/Progetto-SCPA/openMP ludozarr99@160.80.85.52:/data/ludozarr99/Progetto-SCPA

copy-CUDA:
	scp -i /home/ludovico99/.ssh/id_rsa -r /home/ludovico99/Scrivania/Progetto-SCPA/CUDA ludozarr99@160.80.85.52:/data/ludozarr99/Progetto-SCPA

copy-deviceQuery:
	scp -i /home/ludovico99/.ssh/id_rsa -r /home/ludovico99/Scrivania/Progetto-SCPA/CUDA_dev_query ludozarr99@160.80.85.52:/data/ludozarr99/Progetto-SCPA

copy-code:
	scp -i /home/ludovico99/.ssh/id_rsa  -r /home/ludovico99/Scrivania/Progetto-SCPA/ ludozarr99@160.80.85.52:/data/ludozarr99/

copy-make:
	scp -i /home/ludovico99/.ssh/id_rsa  -r /home/ludovico99/Scrivania/Progetto-SCPA/Makefile ludozarr99@160.80.85.52:/data/ludozarr99/Progetto-SCPA/

openmp-csr-compare-serial-parallel:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -D_GNU_SOURCE -DCORRECTNESS conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c -o app

openmp-ellpack-compare-serial-parallel:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -D_GNU_SOURCE -DCORRECTNESS  conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c -o app

openmp-csr-check-conversions:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -D_GNU_SOURCE -DCHECK_CONVERSION  conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c -o app

openmp-ellpack-check-conversions:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -D_GNU_SOURCE -DCHECK_CONVERSION conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c -o app

openmp-csr-serial-samplings:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE  conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c samplings.c -o app

openmp-csr-parallel-samplings:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c samplings.c -o app

openmp-ellpack-serial-samplings:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c samplings.c -o app

openmp-ellpack-parallel-samplings:
	gcc -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c mmio.c main.c utils.c create_mtx_coo.c checks.c samplings.c -o app

matrice_prova:
	./app Matrici/prova.mtx 

adder-dcop-32:
	./app Matrici/adder_dcop_32/adder_dcop_32.mtx 

PR02R:
	./app Matrici/PR02R/PR02R.mtx 

dc1:
	./app Matrici/dc1/dc1.mtx 

raefsky2:
	./app Matrici/raefsky2/raefsky2.mtx 

mhd4800a:
	./app Matrici/mhd4800a/mhd4800a.mtx 

bcsstk17:
	./app Matrici/bcsstk17/bcsstk17.mtx 

amazon0302:
	./app Matrici/amazon0302/amazon0302.mtx 

Cube_Coup_dt0:
	./app Matrici/Cube_Coup_dt0/Cube_Coup_dt0.mtx 

ML_Laplace:
	./app Matrici/ML_Laplace/ML_Laplace.mtx 

clean:
	rm -f *.o
	rm app