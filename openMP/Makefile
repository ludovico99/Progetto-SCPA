serial-CSR_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

parallel-CSR_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DPARALLEL_SAMPLING -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

both-CSR_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DBOTH -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

serial-ELLPACK_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DELLPACK -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

parallel-ELLPACK_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DELLPACK -DPARALLEL_SAMPLING -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP

both-ELLPACK_samping:
	gcc -fopenmp -std=c99 -D_POSIX_SOURCE -DBOTH -DELLPACK -D_GNU_SOURCE serial/serial_product.c serial/conversions.c parallel/conversions.c parallel/parallel_product.c mmio.c main.c -o openMP
