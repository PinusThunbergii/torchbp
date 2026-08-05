#ifndef _TORCHBP_UTIL_CUH
#define _TORCHBP_UTIL_CUH
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define LIBCUDACXX_ENABLE_HOST_NVFP16
// cuda::std::complex multiplication checks for nans which causes
// it to be extremely slow. This can be disabled by definition below on
// new version of the library, but it's not available on the old version
// that is shipped with cuda.
#ifndef LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#define LIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS
#endif
#include <cuda/std/version>
#include <cuda/std/type_traits>
#if _LIBCUDACXX_CUDA_API_VERSION >= 2002000
#  include <cuda/std/complex>
#else
#  include "std_complex.h"
#endif

#include "../util_shared.h"

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

namespace torchbp {

using complex64_t = cuda::std::complex<float>;
using complex32_t = cuda::std::complex<__half>;

// Constant memory for polynomial coefficients
constexpr int POLY_COEF_MAX = 15;
__constant__ float d_poly_coefs[POLY_COEF_MAX];

// Fast asin/atan2 for antenna-pattern angle lookups. The results only index
// the gain table (whose cells are >> 1e-5 rad), so a short polynomial is
// enough: max error ~2e-8 rad (asin, Abramowitz & Stegun 4.4.46) and
// ~2e-6 rad (atan2, degree-11 odd minimax + __fdividef), against ~40-60
// instruction software library implementations.
__device__ static inline float fast_asinf(float x) {
    const float ax = fabsf(x);
    float p =        fmaf(ax, -0.0012624911f, 0.0066700901f);
    p = fmaf(ax, p, -0.0170881256f);
    p = fmaf(ax, p,  0.0308918810f);
    p = fmaf(ax, p, -0.0501743046f);
    p = fmaf(ax, p,  0.0889789874f);
    p = fmaf(ax, p, -0.2145988016f);
    p = fmaf(ax, p,  1.5707963050f);
    const float r = 1.5707963268f - sqrtf(1.0f - ax) * p;
    return copysignf(r, x);
}

__device__ static inline float fast_atan2f(float y, float x) {
    const float ax = fabsf(x);
    const float ay = fabsf(y);
    const float mx = fmaxf(ax, ay);
    const float mn = fminf(ax, ay);
    // z in [0, 1]; atan2(0, 0) = 0 by convention.
    float z = mx == 0.0f ? 0.0f : __fdividef(mn, mx);
    const float z2 = z * z;
    float p =        fmaf(z2, -0.011721630f, 0.052653560f);
    p = fmaf(z2, p, -0.116432027f);
    p = fmaf(z2, p,  0.193542504f);
    p = fmaf(z2, p, -0.332623153f);
    p = fmaf(z2, p,  0.999977233f);
    float a = z * p;
    if (ay > ax) a = 1.5707963268f - a;
    if (x < 0.0f) a = kPI - a;
    return copysignf(a, y);
}

// interp2d, interp2d_grad{x,y}, tx_power_* helpers, lanczos_kernel,
// knab_kernel and knab_kernel_norm are shared with the CPU backend in
// ../util_shared.h.

template<class T, class T2>
__device__ T lanczos_interp_1d(const T2 *img, int n, float pos, int order) {
    float a = 0.5f * order;
    int start = max(0, (int)ceilf(pos - a));
    int end = min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = lanczos_kernel(dx, a);
        T val;
        if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
            half2 val_h = ((half2*)img)[i];
            val = {__half2float(val_h.x), __half2float(val_h.y)};
        } else {
            val = img[i];
        }
        sum += w * val;
    }
    return sum;
}

template<class T, class T2>
__device__ T lanczos_interp_2d(const T2 *img, int nx, int ny, float x, float y, int order) {
    float a = 0.5f * order;
    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = lanczos_kernel(dx, a);
        T row_val = lanczos_interp_1d<T, T2>(img + i * ny, ny, y, order);
        sum += wx * row_val;
    }
    return sum;
}

template<class T, class T2>
__device__ T knab_interp_1d(const T2 *img, int n, float pos, int order, float v, float norm) {
    float a = 0.5f * order;
    int start = max(0, (int)ceilf(pos - a));
    int end = min(n-1, (int)floorf(pos + a));
    T sum{};
    for (int i = start; i <= end; i++) {
        float dx = pos - i;
        float w = knab_kernel(dx, a, v, norm);
        T val;
        if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
            half2 val_h = ((half2*)img)[i];
            val = {__half2float(val_h.x), __half2float(val_h.y)};
        } else {
            val = img[i];
        }
        sum += w * val;
    }
    return sum;
}

template<class T, class T2>
__device__ T knab_interp_2d(const T2 *img, int nx, int ny, float x, float y, int order, float v, float norm) {
    float a = 0.5f * order;
    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel(dx, a, v, norm);
        T row_val = knab_interp_1d<T, T2>(img + i * ny, ny, y, order, v, norm);
        sum += wx * row_val;
    }
    return sum;
}

// 2D Knab interpolation with the x-axis (range) taps demodulated by the
// known local frequency ``fmod_x`` (radians per x-sample): interpolates
// s(n)*exp(-j*fmod_x*(n - x)), which reconstructs s(x) exactly when the
// signal's local spectrum is centered at fmod_x instead of DC. Used by the
// ffbp merges where the subaperture image's range spectrum is shifted by
// the aperture-averaged cos(aspect) projection at close range.
template<class T, class T2>
__device__ T knab_interp_2d_fmod(const T2 *img, int nx, int ny, float x, float y,
                                 int order, float v, float norm, float fmod_x) {
    if (fmod_x == 0.0f) {
        return knab_interp_2d<T, T2>(img, nx, ny, x, y, order, v, norm);
    }
    float a = 0.5f * order;
    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    float rs, rc;
    __sincosf(-fmod_x * (start_x - x), &rs, &rc);
    float ds, dc;
    __sincosf(-fmod_x, &ds, &dc);
    T rot = {rc, rs};
    const T step = {dc, ds};
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel(dx, a, v, norm);
        T row_val = knab_interp_1d<T, T2>(img + i * ny, ny, y, order, v, norm);
        sum += (wx * rot) * row_val;
        rot = rot * step;
    }
    return sum;
}

// 1D windowed-sinc resampler tap. Reads the input signal at continuous
// position ``src`` (in input samples) with a Lanczos kernel. ``cutoff`` <= 1
// lowpasses to ``cutoff`` * input-Nyquist for anti-aliased decimation; it is 1
// for up/equal-rate sampling, giving the plain Lanczos kernel.
template<class T>
__device__ T lanczos_resample_1d(const T *img, int n, float src, int order, float cutoff) {
    float a = 0.5f * order;
    float half = a / cutoff;
    int start = max(0, (int)ceilf(src - half));
    int end = min(n-1, (int)floorf(src + half));
    T sum{};
    for (int i = start; i <= end; i++) {
        float w = cutoff * lanczos_kernel(cutoff * (src - i), a);
        sum += w * img[i];
    }
    return sum;
}

// 1D windowed-sinc resampler tap using the Knab kernel. See
// lanczos_resample_1d for the ``cutoff`` semantics.
template<class T>
__device__ T knab_resample_1d(const T *img, int n, float src, int order, float v, float norm, float cutoff) {
    float a = 0.5f * order;
    float half = a / cutoff;
    int start = max(0, (int)ceilf(src - half));
    int end = min(n-1, (int)floorf(src + half));
    T sum{};
    for (int i = start; i <= end; i++) {
        float w = cutoff * knab_kernel(cutoff * (src - i), a, v, norm);
        sum += w * img[i];
    }
    return sum;
}

// Template-specialized polynomial evaluation for full compile-time unrolling
// Evaluates 1 + c1*x + c2*x^2 + ... + cn*x^n using Horner's method
// N_COEFS is the number of coefficients (c1..cn, excluding implicit c0=1)
template<int N_COEFS>
inline __device__ float polyval_c0_one(float x) {
    static_assert(N_COEFS >= 1 && N_COEFS <= POLY_COEF_MAX, "N_COEFS must be 1-POLY_COEF_MAX");
    float inner = d_poly_coefs[N_COEFS - 1];
    #pragma unroll
    for (int i = N_COEFS - 2; i >= 0; i--) {
        inner = __fmaf_rn(inner, x, d_poly_coefs[i]);
    }
    return __fmaf_rn(x, inner, 1.0f);
}

// Full Knab kernel using polynomial approximation of entire kernel (sinc * window)
// Eliminates sinpif and division - uses ONLY polynomial evaluation with FMA.
// Polynomial is in x²: 1 + c1*x² + c2*x⁴ + ... where x is distance from interpolation point.
// Uses parallel Horner for improved ILP.
// N_COEFS is the number of coefficients (c1..cn, excluding implicit c0=1), must be even
// a2 is the squared half-width (a² where a = order/2)
template<int N_COEFS>
inline __device__ float poly_interp_kernel(float x, float inv_a2) {
    float x2 = x * x;
    // Polynomial was fitted for (x/a)²
    float t = x2 * inv_a2;
    return polyval_c0_one<N_COEFS>(t);
}

template<class T, class T2, int N_COEFS, int MAX_ORDER=8>
__device__ T interp_2d_poly(const T2 *img, int nx, int ny, float x, float y, int order) {
    float a = 0.5f * order;
    float inv_a2 = 1 / (a * a);

    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    int start_y = max(0, (int)ceilf(y - a));
    int end_y = min(ny-1, (int)floorf(y + a));

    int nx_count = end_x - start_x + 1;
    int ny_count = end_y - start_y + 1;

    // Precompute Y weights - polynomial evaluation is pure math, no bounds check needed
    // since start/end already ensure we're within kernel support
    // No point in precomputing wx since they are only used once.
    float wy[MAX_ORDER];

    #pragma unroll
    for (int j = 0; j < MAX_ORDER; j++) {
        if (j < ny_count) {
            float dy = y - (float)(start_y + j);
            wy[j] = poly_interp_kernel<N_COEFS>(dy, inv_a2);
        }
    }

    T sum{};
    #pragma unroll
    for (int i = 0; i < MAX_ORDER; i++) {
        if (i >= nx_count) break;

        float dx = x - (float)(start_x + i);
        float wx = poly_interp_kernel<N_COEFS>(dx, inv_a2);

        const T2 *row = img + (start_x + i) * ny;

        #pragma unroll
        for (int j = 0; j < MAX_ORDER; j++) {
            if (j >= ny_count) break;
            T val;
            if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
                half2 val_h = ((half2*)row)[start_y + j];
                val = {__half2float(val_h.x), __half2float(val_h.y)};
            } else {
                val = row[start_y + j];
            }
            sum += (wx * wy[j]) * val;
        }
    }
    return sum;
}

// interp_2d_poly with the x-axis (range) taps demodulated by ``fmod_x``
// (radians per x-sample); see knab_interp_2d_fmod.
template<class T, class T2, int N_COEFS, int MAX_ORDER=8>
__device__ T interp_2d_poly_fmod(const T2 *img, int nx, int ny, float x, float y,
                                 int order, float fmod_x) {
    if (fmod_x == 0.0f) {
        return interp_2d_poly<T, T2, N_COEFS, MAX_ORDER>(img, nx, ny, x, y, order);
    }
    float a = 0.5f * order;
    float inv_a2 = 1 / (a * a);

    int start_x = max(0, (int)ceilf(x - a));
    int end_x = min(nx-1, (int)floorf(x + a));
    int start_y = max(0, (int)ceilf(y - a));
    int end_y = min(ny-1, (int)floorf(y + a));

    int nx_count = end_x - start_x + 1;
    int ny_count = end_y - start_y + 1;

    float wy[MAX_ORDER];

    #pragma unroll
    for (int j = 0; j < MAX_ORDER; j++) {
        if (j < ny_count) {
            float dy = y - (float)(start_y + j);
            wy[j] = poly_interp_kernel<N_COEFS>(dy, inv_a2);
        }
    }

    float rs, rc;
    __sincosf(-fmod_x * (start_x - x), &rs, &rc);
    float ds, dc;
    __sincosf(-fmod_x, &ds, &dc);
    T rot = {rc, rs};
    const T step = {dc, ds};

    T sum{};
    #pragma unroll
    for (int i = 0; i < MAX_ORDER; i++) {
        if (i >= nx_count) break;

        float dx = x - (float)(start_x + i);
        float wx = poly_interp_kernel<N_COEFS>(dx, inv_a2);

        const T2 *row = img + (start_x + i) * ny;

        T row_sum{};
        #pragma unroll
        for (int j = 0; j < MAX_ORDER; j++) {
            if (j >= ny_count) break;
            T val;
            if constexpr (::cuda::std::is_same_v<T2, complex32_t> || ::cuda::std::is_same_v<T2, half2>) {
                half2 val_h = ((half2*)row)[start_y + j];
                val = {__half2float(val_h.x), __half2float(val_h.y)};
            } else {
                val = row[start_y + j];
            }
            row_sum += wy[j] * val;
        }
        sum += (wx * rot) * row_sum;
        rot = rot * step;
    }
    return sum;
}

}
#endif
