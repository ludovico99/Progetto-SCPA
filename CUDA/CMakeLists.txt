project(prod-sparseMat-denseVec)
cmake_minimum_required(VERSION 2.8)

# Compute capability. See -arch option in nvcc --help.
set (CC 75)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

set(SOURCES
    mmio.c
    parallel/conversions.c
    serial/conversions.c
)

set (CUDA_SOURCES
    main.cu
    parallel/parallel_product.cu
    serial/serial_product.cu    
)

# Divide i file in base all'estensione
# set(CUDA_SOURCES "")
# set(C_SOURCES "")
# foreach(source IN LISTS SOURCES)
#     get_filename_component(extension ${source} EXT)
#     if(${extension} STREQUAL ".cu")
#         list(APPEND CUDA_SOURCES ${source})
#     else()
#         list(APPEND C_SOURCES ${source})
#     endif()
# endforeach()


# target_compile_features(particles PUBLIC cxx_std_11)


# Crea l'eseguibile del progetto
add_executable(cuda ${SOURCES} ${CUDA_SOURCES})

set_target_properties(cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(CMAKE_CXX_FLAGS "-O3")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-DSM_${CC};-arch=sm_${CC};-lineinfo;-Xcompiler=${CMAKE_CXX_FLAGS};-Xptxas=-v)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda PRIVATE OpenMP::OpenMP_CXX)
endif()
