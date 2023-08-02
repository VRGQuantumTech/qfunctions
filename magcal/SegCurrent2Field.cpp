#include<math.h>
#include <stdio.h>

__global__ void SegCurrent2Field(float *Ixrs, int DX, float *Iyrs, int DY,
                                     float *H2D, float *xi, float *yi, float *posx,
                                     float *posy, float *posz, int *loop, int X, int Y, int Z)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int idz = threadIdx.z + blockDim.z * blockIdx.z;
    
    if ((idx < X) && (idy < Y) && (idz < Z))
    {
        float dx = posx[idx + X * idy + X * Y * idz];
        float dy = posy[idx + X * idy + X * Y * idz];
        float dz = posz[idx + X * idy + X * Y * idz];

        float H1 = 0;
        float H2 = 0;
        float H3 = 0;

        for (int i = 0; i < DX * DY; i += 4)
        {
            
            for (int j = 0; j < 4; j++)
            {
                float ijdx =  dx - xi[i + j];
                float ijdy =  dy - yi[i + j];
    
                float modr = sqrtf(ijdx * ijdx + ijdy * ijdy + dz * dz);
                float imodr = 1 / modr;
    
                H1 += Iyrs[i + j] * dz * (imodr * imodr * imodr);
                H2 += -Ixrs[i + j] * dz * (imodr * imodr * imodr);
                H3 += (Ixrs[i + j] * ijdy - Iyrs[i + j] * ijdx) * (imodr * imodr * imodr);

            }
        }
    
    
        if (loop[idx + X * idy + X * Y * idz] == 0)
        {
            H2D[idx + X * idy + X * Y * idz] = H1;
        }
        else if (loop[idx + X * idy + X * Y * idz] == 1)
        {
            H2D[idx + X * idy + X * Y * idz] = H2;
        }
        else if (loop[idx + X * idy + X * Y * idz] == 2)
        {
            H2D[idx + X * idy + X * Y * idz] = H3;
        }    
        
    }
}
