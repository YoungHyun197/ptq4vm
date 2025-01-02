#include <torch/extension.h>

// return output, time vector, q_input, q_weight
torch::Tensor vim_GEMM_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor smooth_scale, torch::Tensor scale_x, torch::Tensor scale_w, int H_size, int qbit);

torch::Tensor vim_GEMM(torch::Tensor x, torch::Tensor w, torch::Tensor smooth_scale, torch::Tensor scale_x, torch::Tensor scale_w, int H_size, int qbit){
    TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
    TORCH_CHECK(x.dim() == 3, "x must be 3D!");

    TORCH_CHECK(w.type().is_cuda(), "w must be a CUDA tensor!");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous!");
    TORCH_CHECK(w.dim() == 2, "w must be 2D!");

    TORCH_CHECK((qbit == 4) || (qbit == 8), "qbit must be 4 or 8!!");
    return vim_GEMM_cuda(x, w, smooth_scale, scale_x, scale_w, H_size, qbit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vim_GEMM", &vim_GEMM);
}