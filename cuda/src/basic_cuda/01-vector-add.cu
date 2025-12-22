
// __global__关键字告诉CUDA编译器这个函数在GPU上运行，但可以从CPU代码调用
__global__ void
vectorAdd(const float *A, const float *B, const float *C, int numElements) {
  int i = blockDim.x * blockIdx.y + threadIdx.x; // 为每个线程计算一个唯一的全局索引

  if (i < numElements)  // 最后一个块可能有超出数组大小的线程
    C[i] = A[i] + B[i];



}
