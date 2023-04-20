# Progetto-SCPA

## Lettura matrici 
Reading a Matrix Market file can be broken into three basic steps:

1. use **mm_read_banner()** to process the 1st line of file and identify the matrix type
2. use a type-specific function, such as **mm_read_mtx_crd_size()**  to skip the optional comments and process the matrix size information
3. use a variant of **scanf()**  to read the numerical data, one matrix entry per line