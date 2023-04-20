all:
	gcc -fopenmp main.c

dot-product-compile-seriale:
	gcc -fopenmp dot-product-ludo.c

dot-product-compile-parallel:
	gcc -fopenmp dot-product-ludo.c -DPARALLEL

timer-compile:
	gcc -fopenmp -std=ec99 -D_POSIX_SOURCE -D_GNU_SOURCE dot-product-luca.c

adder-dcop-32:
	./a.out Matrici/adder_dcop_32/adder_dcop_32.mtx 

