#include "util.h"

// CUDA element-wise interp-combine ops. Mirrors cpu/util.cpp; the per-element
// math is shared with the CPU loop in ../util_shared.h
// (interp2d_combine_elem).
namespace torchbp {

template<typename T, typename T2>
__global__ void interp2d_combine_kernel(
          const T* a,
          const T2* b,
          T* out,
          const int Na0,
          const int Na1,
          const int Nb0,
          const int Nb1,
          const bool is_div) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idbatch = blockIdx.y * blockDim.y + threadIdx.y;
    interp2d_combine_elem<T, T2>(a, b, out, Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
}

static at::Tensor interp2d_combine_cuda(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1,
          bool is_div) {
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
    auto options =
      torch::TensorOptions()
        .dtype(a.dtype())
        .layout(torch::kStrided)
        .device(a.device());
	at::Tensor out = torch::zeros({nbatch, Na0, Na1}, options);

	dim3 thread_per_block = {256, 1};
	// Up-rounding division.
    int blocks = Na0 * Na1;
	unsigned int block_x = (blocks + thread_per_block.x - 1) / thread_per_block.x;
	dim3 block_count = {block_x, static_cast<unsigned int>(nbatch)};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.scalar_type() == at::ScalarType::Float && b.scalar_type() == at::ScalarType::Float) {
        interp2d_combine_kernel<float, float><<<block_count, thread_per_block, 0, stream>>>(
            a_contig.data_ptr<float>(),
            b_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            Na0, Na1, Nb0, Nb1, is_div);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::ComplexFloat) {
        interp2d_combine_kernel<complex64_t, complex64_t><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<const complex64_t*>(b_contig.data_ptr<c10::complex<float>>()),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1, is_div);
    } else if (a.scalar_type() == at::ScalarType::ComplexFloat && b.scalar_type() == at::ScalarType::Float) {
        interp2d_combine_kernel<complex64_t, float><<<block_count, thread_per_block, 0, stream>>>(
            reinterpret_cast<complex64_t*>(a_contig.data_ptr<c10::complex<float>>()),
            b_contig.data_ptr<float>(),
            reinterpret_cast<complex64_t*>(out.data_ptr<c10::complex<float>>()),
            Na0, Na1, Nb0, Nb1, is_div);
    } else {
        AT_ERROR("Unsupported dtype combination for interp2d combine");
    }

	return out;
}

at::Tensor div_2d_interp_linear_cuda(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cuda(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/true);
}

at::Tensor mul_2d_interp_linear_cuda(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cuda(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/false);
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(torchbp, CUDA, m) {
  m.impl("div_2d_interp_linear", &div_2d_interp_linear_cuda);
  m.impl("mul_2d_interp_linear", &mul_2d_interp_linear_cuda);
}

}
