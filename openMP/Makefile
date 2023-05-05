
all-serial:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

all-parallel:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DPARALLEL_SAMPLING -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

matrice_prova:
	./openMP ../Matrici/prova.mtx 

adder-dcop-32:
	./openMP ../Matrici/adder_dcop_32/adder_dcop_32.mtx 

PR02R:
	./openMP ../Matrici/PR02R/PR02R.mtx 

dc1:
	./openMP ../Matrici/dc1/dc1.mtx 

raefsky2:
	./openMP ../Matrici/raefsky2/raefsky2.mtx 

mhd4800a:
	./openMP ../Matrici/mhd4800a/mhd4800a.mtx 

bcsstk17:
	./openMP ../Matrici/bcsstk17/bcsstk17.mtx 

amazon0302:
	./openMP ../Matrici/amazon0302/amazon0302.mtx 

Cube_Coup_dt0:
	./openMP ../Matrici/Cube_Coup_dt0/Cube_Coup_dt0.mtx 

max_nz_by_row:
	cat ../Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^1813 " | wc -l

print_elem_by_row:
	cat ../Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^331 "

copy-all:
	scp -i /home/ludovico99/.ssh/id_rsa -r /home/ludovico99/Scrivania/Progetto-SCPA/openMP ludozarr99@160.80.85.52:/home/ludozarr99/Progetto-SCPA/

copy-file:
	scp -i /home/ludovico99/.ssh/id_rsa  /home/ludovico99/Scrivania/Progetto-SCPA/openMP/Makefile ludozarr99@160.80.85.52:/home/ludozarr99/Progetto-SCPA/openMP