algorithms=("csr" "csr_vector" "csr_vector_by_row" "csr_vector_sub_warp" "csr_adaptive_personalizzato")

sampling_size=10

matrices=("adder_dcop_32" "dc1" "cavity10" "mcfe" "af23560" "raefsky2" "ML_Laplace")

K=(1 3 4 8 12 16 32 64)

module load  gnu mpich cuda

for matrix in "${matrices[@]}";
do  
    for algorithm in "${algorithms[@]}";    
        do  
            make clean

            make all MODE=$algorithm 

        for k in "${K[@]}";
        do 
        
            for curr_sampling in $(eval echo {1..$sampling_size});
            do
                path="Matrici/$matrix/$matrix.mtx"
                ./bin/app "$path" $k
            done

        done

    done 
    
done

