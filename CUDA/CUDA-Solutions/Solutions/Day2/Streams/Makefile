CC = gcc
FC = pgf90
NVCC = nvcc

CFLAG = -O3
FFLAG = -Mcuda=cc35 -fast
NVCCFLAG = -arch=sm_35 -Xcompiler $(CFLAG)

INC = -I $(CUDA_HOME)/include
LIB = -L $(CUDA_HOME)/lib64 -lcuda -lcudart


dummy:
	@echo -e "make Exercise_X_c\tcompile Exercise_X.cu (in CUDA C)"
	@echo -e "make Exercise_X_f\tcompile Exercise_X.cuf (in CUDA FORTRAN)"
	@echo -e "make cuda\t\tcompile all CUDA sources"
	@echo -e "make cudafor\t\tcompile all CUDAFORTRAN sources"
	@echo -e "make clean\t\trm -f *.x *.mod"

cuda: Exercise_1_c Exercise_2_c Exercise_3_c

cudafor: Exercise_1_f Exercise_2_f Exercise_3_f

Exercise_1_c: Exercise_1.cu
	$(NVCC) $(NVCCFLAG) $< -o $@.x $(LIB) $(INC)
Exercise_2_c: Exercise_2.cu
	$(NVCC) $(NVCCFLAG) $< -o $@.x $(LIB) $(INC)
Exercise_3_c: Exercise_3.cu
	$(NVCC) $(NVCCFLAG) $< -o $@.x $(LIB) $(INC)

Exercise_1_f: Exercise_1.cuf
	$(FC) $(FFLAG) $< -o $@.x $(LIB) $(INC)
Exercise_2_f: Exercise_2.cuf
	$(FC) $(FFLAG) $< -o $@.x $(LIB) $(INC)
Exercise_3_f: Exercise_3.cuf
	$(FC) $(FFLAG) $< -o $@.x $(LIB) $(INC)

clean:
	rm -f *.x *.mod
