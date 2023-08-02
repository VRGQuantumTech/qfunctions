#include<math.h>
    
__global__ void ArrayDotCtKernel(float *A, float b, float *C){

//__global__ indicates function that runs on 
//device (GPU) and is called from host (CPU) code

    unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned idy = threadIdx.y + blockDim.y*blockIdx.y;
    unsigned idz = threadIdx.z + blockDim.z*blockIdx.z;
    
    int X = %(DimX)s;
    int Y = %(DimY)s;
    int Z = %(DimZ)s;
                   
    if ( ( idx < X) && (idy < Y) && ( idz < Z) ){
       C[Z*Y*idx + Z*idy + idz] = A[Z*Y*idx + Z*idy + idz]*b;
    }
    
    __syncthreads();
}
