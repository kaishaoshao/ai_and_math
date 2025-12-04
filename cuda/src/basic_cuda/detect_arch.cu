#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
  int device = 0;

  cudaError_t error = cudaSuccess;
  cudaDeviceProp DeviceProp;

  error = cudaGetDeviceCount(&device);
  if (error != cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(error));
    return -1; 
  }

  if(device > 0) {
    fprintf(stderr, "This   ")
  }
 
  return 0;
}
