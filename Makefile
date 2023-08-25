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
#-fmad=false -->>> TO DEBUG

# to choose which implemented algorithms to use
MODE = csr_adaptive_personalizzato
#to do some sampling of the computed stats
SAMPLING = no

OBJS = $(objectsDir)/main.o $(objectsDir)/mmio.o $(objectsDir)/conversions_parallel.o $(objectsDir)/parallel_product_CSR.o $(objectsDir)/parallel_product_ELLPACK.o $(objectsDir)/serial_product.o $(objectsDir)/utils.o $(objectsDir)/create_mtx_coo.o  


ifeq ($(MODE), csr)
    DEFINES = -DCUDA -DCSR
else ifeq ($(MODE), csr_vector) 
	DEFINES = -DCUDA -DCSR -DCSR_VECTOR
else ifeq ($(MODE), csr_vector_by_row) 
	DEFINES = -DCUDA -DCSR -DCSR_VECTOR_BY_ROW
else ifeq ($(MODE), csr_vector_sub_warp) 
	DEFINES = -DCUDA -DCSR -DCSR_VECTOR_SUB_WARP
else ifeq ($(MODE), csr_adaptive_personalizzato)
	DEFINES = -DCUDA -DCSR -DCSR_ADAPTIVE_PERSONALIZZATO
else ifeq ($(MODE), ellpack_sw) 
 	DEFINES = -DCUDA -DELLPACK -DELLPACK_SUB_WARP
else 
	DEFINES = -DCUDA -DELLPACK 
endif

ifeq ($(SAMPLING), yes)
	DEFINES += -DSAMPLINGS
	OBJS += $(objectsDir)/samplings.o
else 
	DEFINES += -DCORRECTNESS
	OBJS += $(objectsDir)/checks.o
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

$(objectsDir)/samplings.o: ./CUDA/samplings.cu
	$(NVCC) $(COMPILER_CUDA) $(DEFINES) $(OPENMP) $(INCLUDES) $(FLAGS)  -o $@ -c $<

$(objectsDir)/serial_product.o: serial_product.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/utils.o: utils.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/checks.o: checks.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(objectsDir)/create_mtx_coo.o: create_mtx_coo.c
	$(NVCC) $(COMPILER_CPP) $(DEFINES) $(OPENMP) -o $@ -c $<

$(binDir)/app: $(OBJS)
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
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE  $(SOURCES) openMP/samplings.c -o $(binDir)/app

openmp-csr-parallel-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DCSR -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE $(SOURCES) openMP/samplings.c -o $(binDir)/app

openmp-ellpack-serial-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_SERIAL -D_GNU_SOURCE $(SOURCES) openMP/samplings.c -o $(binDir)/app

openmp-ellpack-parallel-samplings:
	mkdir -p $(binDir)
	gcc -O3 -fopenmp -std=c99 -DOPENMP -D_POSIX_SOURCE -DELLPACK -DSAMPLINGS -DSAMPLING_PARALLEL -D_GNU_SOURCE $(SOURCES) openMP/samplings.c -o $(binDir)/app

#------------------------------------------------------------------------- RUN ON A SPECIFIED MATRIX ---------------------------------------------------------------------------------------------------

adder-dcop-32:
	$(binDir)/app Matrici/adder_dcop_32/adder_dcop_32.mtx 64

PR02R:
	$(binDir)/app Matrici/PR02R/PR02R.mtx 64

dc1:
	$(binDir)/app Matrici/dc1/dc1.mtx 64

F1:
	$(binDir)/app Matrici/F1/F1.mtx 64

raefsky2:
	$(binDir)/app Matrici/raefsky2/raefsky2.mtx 64

mhd4800a:
	$(binDir)/app Matrici/mhd4800a/mhd4800a.mtx 64

bcsstk17:
	$(binDir)/app Matrici/bcsstk17/bcsstk17.mtx 64

amazon0302:
	$(binDir)/app Matrici/amazon0302/amazon0302.mtx 64

Cube_Coup_dt0:
	$(binDir)/app Matrici/Cube_Coup_dt0/Cube_Coup_dt0.mtx 64

ML_Laplace:
	$(binDir)/app Matrici/ML_Laplace/ML_Laplace.mtx 64

thermal2:
	$(binDir)/app Matrici/thermal2/thermal2.mtx 64

af:
	$(binDir)/app Matrici/af23560/af23560.mtx 64

cavity10:
	$(binDir)/app Matrici/cavity10/cavity10.mtx 64

mcfe:
	$(binDir)/app Matrici/mcfe/mcfe.mtx 64

lung2:
	$(binDir)/app Matrici/lung2/lung2.mtx 64

mhda416:
	$(binDir)/app Matrici/mhda416/mhda416.mtx 64

olm1000:
	$(binDir)/app Matrici/olm1000/olm1000.mtx 64

rdist2:
	$(binDir)/app Matrici/rdist2/rdist2.mtx 64

west2021:
	$(binDir)/app Matrici/west2021/west2021.mtx 64

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
    # DIR_SRC = /home/${USERNAME}/Progetto-SCPA
	DIR_SRC = /home/cap/Scrivania/progetto_finale_SCPA/Progetto-SCPA
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

copy-headers:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/include $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)

copy-make:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/Makefile $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/

copy-main:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/main.c $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/

copy-file:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/utils.c $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/

copy-sampling-scripts:
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/samplings_csr.sh $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/
	scp -i $(SSH_KEY)  -r $(DIR_SRC)/samplings_ellpack.sh $(USERNAME_DEST)@160.80.85.52:$(DIR_DEST)/Progetto-SCPA/