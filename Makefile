binDir=./bin
objectsDir=./obj

#------------------------------------------------------------------------- CUDA ---------------------------------------------------------------------------------------------------------
cudaPath=/usr/local/cuda_save_11.2
CC = 75

COMPILER_CPP=-x c++ -O3 --compiler-options -fpermissive
COMPILER_CUDA=-x cu

NVCC=nvcc
LINK=-lm
OPENMP=--compiler-options -fopenmp
INCLUDES=-I${cudaPath}/samples/common/inc
FLAGS = -DSM_${CC} -arch=sm_${CC} -lineinfo -Xcompiler=-O3 -Xptxas=-v

MODE = csr

ifeq ($(MODE), csr)
    DEFINES= -DCORRECTNESS -DCUDA -DCSR
else
    DEFINES= -DCORRECTNESS -DCUDA -DELLPACK
endif

all: build

build: $(binDir)/app

$(objectsDir)/main.o: main.c
	mkdir -p $(objectsDir)
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/mmio.o: ./lib/mmio.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/conversions_parallel.o: conversions_parallel.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/conversions_serial.o: conversions_serial.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/parallel_product_CSR.o: ./CUDA/parallel_product_CSR.cu
	$(NVCC) $(COMPILER_CUDA) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

$(objectsDir)/parallel_product_ELLPACK.o: ./CUDA/parallel_product_ELLPACK.cu
	$(NVCC) $(COMPILER_CUDA) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

$(objectsDir)/serial_product.o: serial_product.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/utils.o: utils.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/checks.o: checks.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/create_mtx_coo.o: create_mtx_coo.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(binDir)/app: $(objectsDir)/main.o $(objectsDir)/mmio.o $(objectsDir)/conversions_parallel.o $(objectsDir)/parallel_product_CSR.o $(objectsDir)/parallel_product_ELLPACK.o $(objectsDir)/serial_product.o $(objectsDir)/utils.o $(objectsDir)/create_mtx_coo.o
	mkdir -p $(binDir)
	$(NVCC) $(OPENMP) $(DEFINES) $(LINK) $^ -o $@

#------------------------------------------------------------------------- OPENMP ---------------------------------------------------------------------------------------------------------

SOURCES= conversions_parallel.c conversions_serial.c openMP/parallel_product.c serial_product.c lib/mmio.c main.c utils.c create_mtx_coo.c

openmp-csr-compare-serial-parallel:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -D_GNU_SOURCE -DCORRECTNESS $(SOURCES) checks.c -o $(binDir)/app

openmp-ellpack-compare-serial-parallel:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -D_GNU_SOURCE -DCORRECTNESS  $(SOURCES) checks.c -o $(binDir)/app

openmp-csr-check-conversions:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -D_GNU_SOURCE -DCHECK_CONVERSION  $(SOURCES) checks.c -o $(binDir)/app

openmp-ellpack-check-conversions:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -D_GNU_SOURCE -DCHECK_CONVERSION $(SOURCES) checks.c -o $(binDir)/app

openmp-csr-serial-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE  $(SOURCES) samplings.c -o $(binDir)/app

openmp-csr-parallel-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE $(SOURCES) samplings.c -o $(binDir)/app

openmp-ellpack-serial-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE $(SOURCES) samplings.c -o $(binDir)/app

openmp-ellpack-parallel-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE $(SOURCES) samplings.c -o $(binDir)/app

#------------------------------------------------------------------------- RUN ON A SPECIFIED MATRIX ---------------------------------------------------------------------------------------------------

matrice_prova:
	$(binDir)/app Matrici/prova.mtx 

adder-dcop-32:
	$(binDir)/app Matrici/adder_dcop_32/adder_dcop_32.mtx 

PR02R:
	$(binDir)/app Matrici/PR02R/PR02R.mtx 

dc1:
	$(binDir)/app Matrici/dc1/dc1.mtx 

raefsky2:
	$(binDir)/app Matrici/raefsky2/raefsky2.mtx 

mhd4800a:
	$(binDir)/app Matrici/mhd4800a/mhd4800a.mtx 

bcsstk17:
	$(binDir)/app Matrici/bcsstk17/bcsstk17.mtx 

amazon0302:
	$(binDir)/app Matrici/amazon0302/amazon0302.mtx 

Cube_Coup_dt0:
	$(binDir)/app Matrici/Cube_Coup_dt0/Cube_Coup_dt0.mtx 

ML_Laplace:
	$(binDir)/app Matrici/ML_Laplace/ML_Laplace.mtx 

#------------------------------------------------------------------------- DEBUG SCRIPTS ---------------------------------------------------------------------------------------------------------

max_nz_by_row:
	cat Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^1 " | wc -l
#cat ../Matrici/bcsstk17/bcsstk17.mtx | grep "^[0-9]* 4 " | wc -l
print_elem_by_row:
	cat Matrici/bcsstk17/bcsstk17.mtx | grep "^1 "

#------------------------------------------------------------------------- CLEAN ---------------------------------------------------------------------------------------------------------

clean:
	rm -rf $(objectsDir) $(binDir)
	rm -f *.o


#------------------------------------------------------------------------- COPY FILES ---------------------------------------------------------------------------------------------------------
SSH_KEY = /home/${USERNAME}/.ssh/id_rsa

USER = Ludovico

ifeq ($(USER), Ludovico)
    DIR_SRC = /home/${USERNAME}/Scrivania/Progetto-SCPA
	USERNAME_DEST = ludozarr99
	DIR_DEST = /data/ludozarr99
else
    DIR_SRC = /home/${USERNAME}/Progetto-SCPA
	USERNAME_DEST = lcapotombolo
	DIR_DEST = /data/lcapotombolo
endif


copy-openMP:
	scp -i $(SSH_KEY) -r $(DIR_SRC)/openMP $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA

copy-CUDA:
	scp -i $(SSH_KEY) -r $(DIR_SRC)/CUDA $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA

copy-deviceQuery:
	scp -i $(SSH_KEY) -r $(DIR_SRC)/CUDA_dev_query $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA

copy-code:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/ $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)

copy-make:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/Makefile $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/