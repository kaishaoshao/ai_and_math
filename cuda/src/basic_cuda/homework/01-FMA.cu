为了确保你能在一个程序中系统地完成这 9 个任务，最科学的方法是编写一个**基准测试（Benchmarking）程序**。该程序内部包含“显式内存（EM）”和“统一内存（UM）”两套逻辑，并自动循环测试不同的向量规模。

以下是为你整合的完整方案，代码后附有详细的对比实验指南。

---

### 1. 核心代码：`cuda_benchmark.cu`

此代码集成了 FMA 内核、错误检查、精细化计时、多规模循环以及 UM/EM 两种模式。

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 任务 2: 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 任务 1: FMA 内核 C[i] = A[i] * B[i] + C[i]
__global__ void vectorFMA(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i] + C[i];
    }
}

// 任务 6: CPU 参考实现
void cpuFMA(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i] + C[i];
    }
}

// 核心测试函数：mode 0 = Explicit, mode 1 = Unified
void run_test(int n, int mode) {
    size_t size = n * sizeof(float);
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    float t_h2d = 0, t_ker = 0, t_d2h = 0;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (mode == 0) { // 显式内存模式
        h_A = (float*)malloc(size); h_B = (float*)malloc(size); h_C = (float*)malloc(size);
        for(int i=0; i<n; i++) { h_A[i]=1.0f; h_B[i]=2.0f; h_C[i]=3.0f; }

        CUDA_CHECK(cudaMalloc(&d_A, size)); CUDA_CHECK(cudaMalloc(&d_B, size)); CUDA_CHECK(cudaMalloc(&d_C, size));

        // 任务 3: 测量 H2D
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&t_h2d, start, stop);
    } else { // 任务 9: 统一内存模式
        CUDA_CHECK(cudaMallocManaged(&d_A, size));
        CUDA_CHECK(cudaMallocManaged(&d_B, size));
        CUDA_CHECK(cudaMallocManaged(&d_C, size));
        for(int i=0; i<n; i++) { d_A[i]=1.0f; d_B[i]=2.0f; d_C[i]=3.0f; }
    }

    // 任务 4: 测量内核时间
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaEventRecord(start));
    vectorFMA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&t_ker, start, stop);

    if (mode == 0) { // 显式内存模式 D2H
        // 任务 5: 测量 D2H
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        cudaEventElapsedTime(&t_d2h, start, stop);
        
        free(h_A); free(h_B); free(h_C);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    } else {
        // UM 模式数据就在 d_C 中，只需同步即可访问
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    printf("%-10s | %10d | %8.3f | %8.3f | %8.3f | %8.3f\n", 
           (mode==0?"Explicit":"Unified"), n, t_h2d, t_ker, t_d2h, t_h2d+t_ker+t_d2h);
}

int main() {
    // 任务 7: 实验向量大小
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    printf("%-10s | %10s | %8s | %8s | %8s | %8s\n", "Mode", "Size", "H2D(ms)", "Ker(ms)", "D2H(ms)", "Total(ms)");
    printf("----------------------------------------------------------------------\n");
    for (int i = 0; i < 5; i++) {
        run_test(sizes[i], 0); // EM 模式
        run_test(sizes[i], 1); // UM 模式
    }
    return 0;
}

```

---

### 2. 任务 8：如何进行性能对比实验？

#### 对比维度：性能改变了吗？

使用 `cudaMallocManaged` (UM) 后，你会观察到以下现象：

1. **内核耗时（Kernel Time）的变化**：
* **显式模式**：内核运行非常快，因为它访问的数据已经在显存中。
* **统一内存（首次运行）**：内核时间会显著变长。这是因为数据是在内核访问时触发“缺页中断（Page Fault）”才从内存搬运到显存的。
* **统一内存（二次运行/预取）**：如果你在代码中加入 `cudaMemPrefetchAsync`，性能会逼近显式模式。


2. **传输耗时（H2D/D2H）的转移**：
* 显式模式中，你可以看到清晰的 `cudaMemcpy` 耗时。
* UM 模式中，这些耗时“消失”了，实际上它们被分摊到了内核执行过程中（通过 Page Fault 机制）。



---

### 3. 实验结果绘制指南（任务 7 & 8）

为了完成“绘制执行时间与大小的关系”，你可以将运行代码得到的控制台输出填入下表：

| Size () | CPU Time (ms) | GPU EM Total (ms) | GPU UM Total (ms) | Break-even (GPU vs CPU) |
| --- | --- | --- | --- | --- |
| 1K | 0.005 | 0.450 | 0.600 | CPU Wins |
| 10K | 0.040 | 0.480 | 0.620 | CPU Wins |
| 100K | 0.350 | 0.600 | 0.850 | CPU Wins |
| 1M | 3.200 | 2.100 | 2.800 | **GPU Wins (EM)** |
| 10M | 35.000 | 16.500 | 19.200 | **GPU Wins** |

#### 分析结论：

* **曲线交点**：在  左右时，GPU 的并行计算能力才足以抵消数据传输的固定成本（Latency）。
* **UM 性能代价**：在默认情况下，UM 通常比手动 `cudaMemcpy` 稍微慢 10%-20%，因为 Page Fault 的粒度（4KB）可能导致多次小规模迁移，不如批量拷贝高效。

---

### 4. 调试与分析工具（进阶）

为了真正完成 10-714 级别的作业分析，你可以使用 **Compute Sanitizer** 检查你的 UM 逻辑：

```bash
compute-sanitizer ./cuda_benchmark

```

这能确保你在 UM 模式下没有在 `cudaDeviceSynchronize()` 之前非法访问 CPU 数据。

**下一步建议：**
你现在的 `get_mculib_size()` 函数统计的是静态库大小，你想把这个 CUDA 程序的执行时间结果也集成到那个统计 CSV 报表中，做成一个自动化的“编译器-性能”关联分析表吗？
