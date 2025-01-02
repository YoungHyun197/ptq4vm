#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"

cudaError_t cutlass_strided_batched_sgemm_int8(
  int m, 
  int n,
  int k,
  const int8_t *A,
  int lda,
  long long int batch_stride_A,
  const int8_t *B,
  int ldb,
  long long int batch_stride_B,
  int32_t *C,
  int ldc,
  long long int batch_stride_C,
  int batch_count) {

  using ElementComputeEpilogue = int32_t;
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); // alpha = 1
  ElementComputeEpilogue beta = ElementComputeEpilogue(0); // beta = 0 (default GEMM config)

  using Gemm = cutlass::gemm::device::GemmBatched<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlass_strided_batched_sgemm_int4(
  int m, 
  int n,
  int k,
  const int8_t *A, // const cutlass::int4b_t *A
  int lda,
  long long int batch_stride_A,
  const int8_t *B, // const cutlass::int4b_t *B
  int ldb,
  long long int batch_stride_B,
  int32_t *C,
  int ldc,
  long long int batch_stride_C,
  int batch_count) {

  using ElementComputeEpilogue = int32_t;
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1); // alpha = 1
  ElementComputeEpilogue beta = ElementComputeEpilogue(0); // beta = 0 (default GEMM config)

  using Gemm = cutlass::gemm::device::GemmBatched<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

// activation smoothing and quantization - per tensor
template<typename scalar_t>
__global__ void act_smq_per_tensor(scalar_t * MatI, int8_t * MatO, scalar_t * smooth_scales, scalar_t * quant_scales, int qbit){
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Calculate indices with padding
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.x + ty;
  int Col = bx * blockDim.y + tx;

  int sdataIdx = ty * blockDim.x + tx;
  int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

  // Upload each block of MatI to sdata, considering padding
  if (threadIdx.x < blockDim.x) {
      sdata[sdataIdx] = MatI[matIdx];
  }
  __syncthreads();

  // Quantization
  scalar_t smooth_scale = smooth_scales[Col];
  scalar_t quant_scale = quant_scales[0];
  if (qbit == 8){
    sdata[sdataIdx] = std::clamp((int)(round((sdata[sdataIdx] / (quant_scale * smooth_scale)))), -127, 127);
  }
  else if (qbit == 4) {
    sdata[sdataIdx] = std::clamp((int)(round((sdata[sdataIdx] / (quant_scale * smooth_scale)))), -7, 7);
  }
  __syncthreads();

  // Download sdata to each block of MatI, considering padding and packing
  if (threadIdx.x < blockDim.x) {
    if (qbit == 8){
      MatO[matIdx] = sdata[sdataIdx];
    }
    else if (qbit == 4) {
      MatO[matIdx] = ((int)sdata[(sdataIdx<<1)+1] << 4) | ((int)sdata[sdataIdx<<1] & 15);
    }
  }
  __syncthreads();
}

// activation smoothing and quantization - per token
template<typename scalar_t>
__global__ void act_smq_per_token(scalar_t * MatI, int8_t * MatO, scalar_t * smooth_scales, scalar_t * quant_scales, int qbit){
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Calculate indices with padding
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.x + ty;
  int Col = bx * blockDim.y + tx;

  int sdataIdx = ty * blockDim.x + tx;
  int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

  // Upload each block of MatI to sdata, considering padding
  if (threadIdx.x < blockDim.x) {
      sdata[sdataIdx] = MatI[matIdx];
  }
  __syncthreads();

  // Quantization
  scalar_t smooth_scale = smooth_scales[Col];
  scalar_t quant_scale = quant_scales[Row];
  if (qbit == 8){
    sdata[sdataIdx] = std::clamp((int)(round((sdata[sdataIdx] / (quant_scale * smooth_scale)))), -127, 127);
  }
  else if (qbit == 4) {
    sdata[sdataIdx] = std::clamp((int)(round((sdata[sdataIdx] / (quant_scale * smooth_scale)))), -7, 7);
  }
  __syncthreads();

  // Download sdata to each block of MatI, considering padding and packing
  if (threadIdx.x < blockDim.x) {
    if (qbit == 8){
      MatO[matIdx] = sdata[sdataIdx];
    }
    else if (qbit == 4) {
      MatO[matIdx] = ((int)sdata[(sdataIdx<<1)+1] << 4) | ((int)sdata[sdataIdx<<1] & 15);
    }
  }
  __syncthreads();
}

// dequant - per tensor
template<typename scalar_t>
__global__ void dequantize_per_tensor(const int32_t * gemm, scalar_t * __restrict__ output, scalar_t * s){
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Calculate indices with padding
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int sdataIdx = ty * blockDim.x + tx;
  int matIdx = by * gridDim.x + bx * blockDim.x * blockDim.y + ty * blockDim.x + tx;

  // Upload each block of MatI to sdata, considering padding
  if (threadIdx.x < blockDim.x) {
      sdata[sdataIdx] = gemm[matIdx];
  }
  __syncthreads();
  
  // Dequantization
  sdata[sdataIdx] = s[0] * sdata[sdataIdx];
  __syncthreads();

  // Download sdata to each block of output
  if (threadIdx.x < blockDim.x) {
      output[matIdx] = sdata[sdataIdx];
  }
  __syncthreads();
}

// dequant - per token
template<typename scalar_t>
__global__ void dequantize_per_token(const int32_t * gemm, scalar_t * __restrict__ output, scalar_t * s){
  extern __shared__ char smem[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

  // Calculate indices with padding

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int sdataIdx = ty * blockDim.x + tx;
  int sdataIdx_2 = ((blockDim.y-1) * blockDim.x + (blockDim.x-1)) + ty * blockDim.x + tx;
  int matIdx = bx * blockDim.x * blockDim.y + ty * blockDim.y + tx;

  // Upload each block of MatI to sdata, considering padding
  if (threadIdx.x < blockDim.x) {
      sdata[sdataIdx] = gemm[matIdx];
      sdata[sdataIdx_2] = s[matIdx];
  }
  __syncthreads();
  
  // Dequantization
  sdata[sdataIdx] = s[sdataIdx_2] * sdata[sdataIdx];
  __syncthreads();

  // Download sdata to each block of output
  if (threadIdx.x < blockDim.x) {
      output[matIdx] = sdata[sdataIdx];
  }
  __syncthreads();
}

torch::Tensor vim_GEMM_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor smooth_scale, torch::Tensor scale_x, torch::Tensor scale_w, int H_size, int qbit){
  cudaError_t result;

  //x = (batch_count, m, k): fp16, w = (n, k): int8, GEMM = (batch_count, m, n): int32
  //For int4, w = (n, k//2): int8
  //smooth_scale = (1, k): fp16, scale_x = (m, 1): fp16, scale_w = (n, 1): fp16 - per token
  //smooth_scale = (1, k): fp16, scale_x = (1, 1): fp16, scale_w = (1, 1): fp16 - per tensor
  long long int m = x.size(1);
  long long int k = x.size(2);
  long long int n = w.size(0);
  long long int batched_count = x.size(0);
  long long int x_height = batched_count * m;
  int block_height = H_size;
  int block_width = 1024/H_size;
  
  torch::Tensor x_colm = x.view({batched_count * m, k}).contiguous();

  // Ready for shared_memory
  size_t shared_memory_bytes = 96 * 1024;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    cudaFuncSetAttribute(
        act_smq_per_tensor<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    cudaFuncSetAttribute(
        act_smq_per_token<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    cudaFuncSetAttribute(
        dequantize_per_tensor<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    cudaFuncSetAttribute(
        dequantize_per_token<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes);
    }));

  dim3 grid_size((x_height + block_height - 1) / block_height, (k + block_width - 1) / block_width);
  dim3 block_size(block_width);
  auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
  torch::Tensor gemm = torch::empty({batched_count * m, n}, option_gemm);
  
  if (qbit == 8){
    // activation smoothing and quantization
    auto option_x_q = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    torch::Tensor x_q = torch::empty({x_height, k}, option_x_q);
    
    if (scale_x.numel() == 1){
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_tensor<scalar_t><<<grid_size, block_size, shared_memory_bytes>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), qbit);
      }));
    }
    else{
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_token<scalar_t><<<grid_size, block_size, shared_memory_bytes>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), qbit);
      }));
    }  

    // GEMM
    int const lda = k;
    int const ldb = k;
    int const ldc = n;

    long long int batch_stride_A = static_cast<long long int>(lda) * m;
    long long int batch_stride_B = 0;
    long long int batch_stride_C = static_cast<long long int>(ldc) * m;
    
    result = cutlass_strided_batched_sgemm_int8(m, n, k, 
            x_q.data_ptr<int8_t>(), lda, batch_stride_A,
            w.data_ptr<int8_t>(), ldb, batch_stride_B,
            gemm.data_ptr<int32_t>(), ldc, batch_stride_C, batched_count);
  }
  else if (qbit == 4) {
    // activation smoothing and quantization
    long long int k_int4 = k>>1;

    auto option_x_q = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    torch::Tensor x_q = torch::empty({x_height, k_int4}, option_x_q);

    if (scale_x.numel() == 1){
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_tensor<scalar_t><<<grid_size, block_size, shared_memory_bytes>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), qbit);
      }));
    }
    else{
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
      act_smq_per_token<scalar_t><<<grid_size, block_size, shared_memory_bytes>>>(
          x_colm.data_ptr<scalar_t>(), x_q.data_ptr<int8_t>(), smooth_scale.data_ptr<scalar_t>(), scale_x.data_ptr<scalar_t>(), qbit);
      }));
    }

    // GEMM
    int const lda = k_int4;
    int const ldb = k_int4;
    int const ldc = n;

    long long int batch_stride_A = static_cast<long long int>(lda) * m;
    long long int batch_stride_B = 0;
    long long int batch_stride_C = static_cast<long long int>(ldc) * m;
    
    // std::cout << "x_q size: ";
    // for (int i = 0; i < x_q.dim(); i++) {
    //     std::cout << x_q.size(i);
    //     if (i < x_q.dim() - 1) {
    //         std::cout << " x ";
    //     }
    // }
    // std::cout << std::endl;

    // std::cout << "w size: ";
    // for (int i = 0; i < w.dim(); i++) {
    //     std::cout << w.size(i);
    //     if (i < w.dim() - 1) {
    //         std::cout << " x ";
    //     }
    // }
    // std::cout << std::endl;

    result = cutlass_strided_batched_sgemm_int4(m, n, k_int4, 
            x_q.data_ptr<int8_t>(), lda, batch_stride_A,
            w.data_ptr<int8_t>(), ldb, batch_stride_B,
            gemm.data_ptr<int32_t>(), ldc, batch_stride_C, batched_count);
  }
  
  // Dequant
  auto option_dequant = torch::TensorOptions().dtype(x.dtype()).device(x.device());
  torch::Tensor y = torch::empty({batched_count * m, n}, option_dequant);

  dim3 grid_size_dequant((x_height + block_height - 1) / block_height, (n + block_width - 1) / block_width);

  if (scale_x.numel() == 1){
    torch::Tensor final_scale = scale_x * scale_w;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    dequantize_per_tensor<scalar_t><<<grid_size_dequant, block_size, shared_memory_bytes>>>(
        gemm.data_ptr<int32_t>(), 
        y.data_ptr<scalar_t>(),
        final_scale.data_ptr<scalar_t>());
    }));
  }
  else{
    torch::Tensor final_scale = torch::matmul(scale_x, scale_w.transpose(0,1));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "vim_GEMM_cuda", ([&] {
    dequantize_per_token<scalar_t><<<grid_size_dequant, block_size, shared_memory_bytes>>>(
        gemm.data_ptr<int32_t>(), 
        y.data_ptr<scalar_t>(),
        final_scale.data_ptr<scalar_t>());
    }));
  }

  return y.view({batched_count, m, n}).contiguous();

}