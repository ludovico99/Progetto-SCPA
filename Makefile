all:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -D_GNU_SOURCE main.c

matrice_prova:
	./a.out Matrici/prova.mtx 

dot-product-compile-seriale:
	gcc -fopenmp dot-product-ludo.c

dot-product-compile-parallel:
	gcc -fopenmp dot-product-ludo.c -DPARALLEL

timer-compile:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -D_GNU_SOURCE dot-product-ludo.c

adder-dcop-32:
	./a.out Matrici/adder_dcop_32/adder_dcop_32.mtx 

PR02R:
	./a.out Matrici/PR02R/PR02R.mtx 

dc1:
	./a.out Matrici/dc1/dc1.mtx 

raefsky2:
	./a.out Matrici/raefsky2/raefsky2.mtx 

mhd4800a:
	./a.out Matrici/mhd4800a/mhd4800a.mtx 

bcsstk17:
	./a.out Matrici/bcsstk17/bcsstk17.mtx 

amazon0302:
	./a.out Matrici/amazon0302/amazon0302.mtx 

max_nz_by_row:
	cat Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^1813 " | wc -l

print_elem_by_row:
	cat Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^331 "

copy-all:
	scp -i /home/ludovico99/.ssh/id_rsa -r /home/ludovico99/Scrivania/Progetto-SCPA ludozarr99@160.80.85.52:/home/ludozarr99

copy-file:
	scp -i /home/ludovico99/.ssh/id_rsa  /home/ludovico99/Scrivania/Progetto-SCPA/Makefile ludozarr99@160.80.85.52:/home/ludozarr99/Progetto-SCPA