#include "util.h"

// CPU element-wise interp-combine ops. Mirrors cuda/util.cu; the per-element
// math is shared with the CUDA kernel in ../util_shared.h
// (interp2d_combine_elem).
namespace torchbp {

static at::Tensor interp2d_combine_cpu(
          const at::Tensor &a,
          const at::Tensor &b,
          int64_t nbatch,
          int64_t Na0,
          int64_t Na1,
          int64_t Nb0,
          int64_t Nb1,
          bool is_div) {
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor out = torch::zeros({nbatch, Na0, Na1}, a_contig.options());

    // See backprojection_cart_2d_cpu for why the team size is set explicitly.
    omp_set_num_threads(omp_get_num_procs());

    const bool a_complex = a.scalar_type() == at::ScalarType::ComplexFloat;
    const bool b_complex = b.scalar_type() == at::ScalarType::ComplexFloat;

#pragma omp parallel for collapse(2)
    for (int idbatch = 0; idbatch < nbatch; idbatch++) {
        for (int idx = 0; idx < Na0 * Na1; idx++) {
            if (a_complex && b_complex) {
                interp2d_combine_elem<complex64_t, complex64_t>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        (const complex64_t*)b_contig.data_ptr<c10::complex<float>>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (a_complex && !b_complex) {
                interp2d_combine_elem<complex64_t, float>(
                        (const complex64_t*)a_contig.data_ptr<c10::complex<float>>(),
                        b_contig.data_ptr<float>(),
                        (complex64_t*)out.data_ptr<c10::complex<float>>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else if (!a_complex && !b_complex) {
                interp2d_combine_elem<float, float>(
                        a_contig.data_ptr<float>(),
                        b_contig.data_ptr<float>(),
                        out.data_ptr<float>(),
                        Na0, Na1, Nb0, Nb1, is_div, idx, idbatch);
            } else {
                AT_ERROR("Unsupported dtype combination for interp2d combine");
            }
        }
    }
    return out;
}

at::Tensor div_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/true);
}

at::Tensor mul_2d_interp_linear_cpu(
          const at::Tensor &a, const at::Tensor &b,
          int64_t nbatch, int64_t Na0, int64_t Na1, int64_t Nb0, int64_t Nb1) {
    return interp2d_combine_cpu(a, b, nbatch, Na0, Na1, Nb0, Nb1, /*is_div=*/false);
}

// Registers CPU implementations
TORCH_LIBRARY_IMPL(torchbp, CPU, m) {
  m.impl("div_2d_interp_linear", &div_2d_interp_linear_cpu);
  m.impl("mul_2d_interp_linear", &mul_2d_interp_linear_cpu);
}

}
