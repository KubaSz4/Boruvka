#include "cuda.h"
#include <cstdio>
#include <climits>
#include <algorithm>


#define THREADS 1024
#define TRANSPOSE_THREADS 4

const char *cuda_error_string(CUresult result) 
{ 
    switch(result) { 
    case CUDA_SUCCESS: return "No errors"; 
    case CUDA_ERROR_INVALID_VALUE: return "Invalid value"; 
    case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory"; 
    case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized"; 
    case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized"; 

    case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available"; 
    case CUDA_ERROR_INVALID_DEVICE: return "Invalid device"; 

    case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image"; 
    case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context"; 
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current"; 
    case CUDA_ERROR_MAP_FAILED: return "Map failed"; 
    case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed"; 
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped"; 
    case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped"; 
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU"; 
    case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired"; 
    case CUDA_ERROR_NOT_MAPPED: return "Not mapped"; 
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Mapped resource not available for access as an array"; 
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Mapped resource not available for access as a pointer"; 
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "Uncorrectable ECC error detected"; 
    case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUlimit not supported by device";    

    case CUDA_ERROR_INVALID_SOURCE: return "Invalid source"; 
    case CUDA_ERROR_FILE_NOT_FOUND: return "File not found"; 
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Link to a shared object failed to resolve"; 
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Shared object initialization failed"; 

    case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle"; 

    case CUDA_ERROR_NOT_FOUND: return "Not found"; 

    case CUDA_ERROR_NOT_READY: return "CUDA not ready"; 

    case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed"; 
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources"; 
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout"; 
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing"; 

    case CUDA_ERROR_UNKNOWN: return "Unknown error"; 

    default: return "Unknown CUDA error value"; 
    } 
}

inline void run_cuFunction(CUresult res, const char* message){
    if (res != CUDA_SUCCESS){
        printf("%s %d - %s\n", message, res, cuda_error_string(res)); 
        exit(1);
    }
}

inline void swap(CUdeviceptr & A, CUdeviceptr & B){
    CUdeviceptr tmp = A;
    A = B;
    B = tmp;
}

inline int min(int a, int b){
    return (a<b ? a : b);
}

void print(int* T, int n, const char* name){
    printf("%s:\n", name);
    for(int i=0; i<n; i++){
        printf("%10d ", T[i]);
    }
    printf("\n");
}

void print(int* T, int n, int m, const char* name){
    printf("%s:\n", name);
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            printf("%10d ", T[i*m+j]);
        }
        printf("\n");
    }
}

void print(CUdeviceptr GDev, CUdeviceptr parentDev, CUdeviceptr newParentDev, CUdeviceptr MSTresDev, CUdeviceptr roundNeighbourDistDev, int n){
    CUresult res;

    int * tmp = (int*) malloc(sizeof(int)*n*n);

    run_cuFunction(cuMemHostRegister(tmp, sizeof(int)*n*n, 0), "cannot register");

    run_cuFunction(cuMemcpyDtoH(tmp, GDev, sizeof(int)*n*n), "cannot copy DtoH");
    print(tmp, n, n, "G");

    run_cuFunction(cuMemcpyDtoH(tmp, parentDev, sizeof(int)*n), "cannot copy DtoH");
    print(tmp, n, "parent");

    run_cuFunction(cuMemcpyDtoH(tmp, newParentDev, sizeof(int)*n), "cannot copy DtoH");
    print(tmp, n, "newParent");

    run_cuFunction(cuMemcpyDtoH(tmp, roundNeighbourDistDev, sizeof(int)*n), "cannot copy DtoH");
    print(tmp, n, "roundNeighbourDist");

    run_cuFunction(cuMemcpyDtoH(tmp, MSTresDev, sizeof(int)*n), "cannot copy DtoH");
    print(tmp, n, "MSTres");
}




int boruvka(int n, int * G){
    cuInit(0);
    
    CUdevice cuDevice;
    run_cuFunction(cuDeviceGet(&cuDevice, 0), "cannot acquire device 0");
    
    CUcontext cuContext;
    run_cuFunction(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");

    CUmodule cuModule = (CUmodule)0;
    run_cuFunction(cuModuleLoad(&cuModule, "boruvka.ptx"), "cannot load module"); 

    CUfunction set;
    run_cuFunction(cuModuleGetFunction(&set, cuModule, "set"), "cannot acquire kernel handle");

    CUfunction closest_neighbour;
    run_cuFunction(cuModuleGetFunction(&closest_neighbour, cuModule, "closest_neighbour"), "cannot acquire kernel handle");
    
    CUfunction unionn;
    run_cuFunction(cuModuleGetFunction(&unionn, cuModule, "unionn"), "cannot acquire kernel handle");

    CUfunction subtract;
    run_cuFunction(cuModuleGetFunction(&subtract, cuModule, "subtract"), "cannot acquire kernel handle");

    CUfunction transpose;
    run_cuFunction(cuModuleGetFunction(&transpose, cuModule, "transpose"), "cannot acquire kernel handle");

    CUfunction merge;
    run_cuFunction(cuModuleGetFunction(&merge, cuModule, "merge"), "cannot acquire kernel handle");

    int threads_X = min(THREADS, n);
    int blocks_X = n/threads_X;
    if (n%threads_X)
        blocks_X++;


    int threads_transpose_X = min(TRANSPOSE_THREADS, n);

    //threads_transpose_X must be a factor of n
    while(n%threads_transpose_X) threads_transpose_X--;

    int blocks_transpose_X = n/threads_transpose_X;

    CUdeviceptr GDev;
    CUdeviceptr GDevT;
    CUdeviceptr parentDev;
    CUdeviceptr newParentDev;
    CUdeviceptr MSTresDev;
    CUdeviceptr roundNeighbourDistDev;
    CUdeviceptr toSubtractDev;

    run_cuFunction(cuMemAlloc(&GDev, sizeof(int)*n*n), "cannot alloc G");

    run_cuFunction(cuMemAlloc(&GDevT, sizeof(int)*n*n), "cannot alloc G");

    run_cuFunction(cuMemAlloc(&parentDev, sizeof(int) *n), "cannot alloc parent");

    run_cuFunction(cuMemAlloc(&newParentDev, sizeof(int) *n), "cannot alloc parent");

    run_cuFunction(cuMemAlloc(&MSTresDev, sizeof(int) *n), "cannot alloc res");
    
    run_cuFunction(cuMemAlloc(&roundNeighbourDistDev, sizeof(int) *n), "cannot alloc res");

    run_cuFunction(cuMemAlloc(&toSubtractDev, sizeof(int) *n), "cannot alloc res");

    void* args[] = {&parentDev, &newParentDev, &MSTresDev, &n};
    run_cuFunction(cuLaunchKernel(set, blocks_X, 1, 1, threads_X, 1, 1, 0, 0, args, 0), "cannot run kernel");

    run_cuFunction(cuMemHostRegister(G, sizeof(int)*n*n, 0), "cannot register");

    run_cuFunction(cuMemcpyHtoD(GDev, G, sizeof(int)*n*n), "cannot copy HtoD");

    int * goOn = (int*) malloc(sizeof(int));
    *goOn = 1;

    run_cuFunction(cuMemHostRegister(goOn, sizeof(int), 0), "cannot register");

    CUdeviceptr goOnDev;
    run_cuFunction(cuMemAlloc(&goOnDev, sizeof(int)), "cannot alloc");

    while(true){ 
//        print(GDev, parentDev, newParentDev, MSTresDev, roundNeighbourDistDev, n);    
        *goOn = 0;
        run_cuFunction(cuMemcpyHtoD(goOnDev, goOn, sizeof(int)), "cannot copy HtoD");

        //find the closest neighbour from other component for each vertex
        void* args1[] = {&GDev, &parentDev, &newParentDev, &roundNeighbourDistDev, &toSubtractDev, &n};
        run_cuFunction(cuLaunchKernel(closest_neighbour, blocks_X, 1, 1, threads_X, 1, 1, 0, 0, args1, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");

//        print(GDev, parentDev, newParentDev, MSTresDev, roundNeighbourDistDev, n);

        //union vertex that were connected by neighbours from previous step
        void* args2[] = {&parentDev, &newParentDev, &MSTresDev, &roundNeighbourDistDev, &toSubtractDev, &n, &goOnDev};
        run_cuFunction(cuLaunchKernel(unionn, blocks_X, 1, 1, threads_X, 1, 1, 0, 0, args2, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");
        
        //substract the repeated edge from each new vertex (component)
        void* args6[] = {&newParentDev, &MSTresDev, &toSubtractDev, &n};
        run_cuFunction(cuLaunchKernel(subtract, blocks_X, 1, 1, threads_X, 1, 1, 0, 0, args6, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");

        run_cuFunction(cuMemcpyDtoH(goOn, goOnDev, sizeof(int)), "cannot copy DtoH");

//        print(GDev, parentDev, newParentDev, MSTresDev, roundNeighbourDistDev, n);

        if(!*goOn) break;

        void* args3[] = {&GDev, &GDevT, &n};
        run_cuFunction(cuLaunchKernel(transpose, blocks_transpose_X, blocks_transpose_X, 1, threads_transpose_X, threads_transpose_X, 1, 0, 0, args3, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");

        //merge graph for new components
        void* args4[] = {&GDevT, &parentDev, &n};
        run_cuFunction(cuLaunchKernel(merge, blocks_X, 1, 1, threads_X, 1, 1, 0, 0, args4, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");

        void* args5[] = {&GDevT, &GDev, &n};
        run_cuFunction(cuLaunchKernel(transpose, blocks_transpose_X, blocks_transpose_X, 1, threads_transpose_X, threads_transpose_X, 1, 0, 0, args5, 0), "cannot run kernel");
        run_cuFunction(cuCtxSynchronize(), "cannot sync");

    }
    
    int RES;
    
    run_cuFunction(cuMemcpyDtoH(&RES, MSTresDev, sizeof(int)), "cannot copy DtoH");

    return RES;
}