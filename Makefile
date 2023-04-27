all:
	gcc -fopenmp main.c

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

max_nz_by_row:
	cat Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^1813 " | wc -l

print_elem_by_row:
	cat Matrici/adder_dcop_32/adder_dcop_32.mtx | grep "^331 "
