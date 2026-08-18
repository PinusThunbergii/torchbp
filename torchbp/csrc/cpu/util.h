#ifndef _TORCHBP_CPU_UTIL_H
#define _TORCHBP_CPU_UTIL_H
// Shared CPU helpers (complex operators, lanczos / knab / polynomial
// interpolation). CPU analogue of cuda/util.h. Backend-independent helpers
// (constants, interp2d, sincospi, the kernel functions, tx_power_*) live in
// ../util_shared.h.
#include <ATen/Operators.h>
#include <ATen/ops/fft_ifft.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <map>
#include <mutex>
#include <tuple>
#include <omp.h>

namespace torchbp {

// Define mixed float * c10::complex<double> multiplication. c10 only provides
// complex<T> * T.
// These have to be declared before ../util_shared.h is included.
inline c10::complex<double> operator * (const float &a, const c10::complex<double> &b){
    return c10::complex<double>(b.real() * (double)a, b.imag() * (double)a);
}

inline c10::complex<double> operator * (const c10::complex<double> &b, const float &a){
    return c10::complex<double>(b.real() * (double)a, b.imag() * (double)a);
}

}

#include "../util_shared.h"

namespace torchbp {

using complex64_t = c10::complex<float>;

// Branchless float asin (cephes single-precision coefficients), a few ulp
// from libm asinf. No calls or branches, so it vectorizes when inlined into
// SIMD loops. libm asinf blocks vectorization. |x| > 1 returns NaN.
static inline float asinf_fast(float x) {
    const float a = fabsf(x);
    const bool big = a > 0.5f;
    const float z1 = 0.5f * (1.0f - a);
    const float z = big ? z1 : a * a;
    const float w = big ? sqrtf(z1) : a;
    float p = 4.2163199048e-2f;
    p = p * z + 2.4181311049e-2f;
    p = p * z + 4.5470025998e-2f;
    p = p * z + 7.4953002686e-2f;
    p = p * z + 1.6666752422e-1f;
    p = p * z * w + w;
    p = big ? 0.5f * kPI - 2.0f * p : p;
    return copysignf(p, x);
}

// Branchless float atan2 (cephes single-precision polynomial), a few ulp from
// libm atan2f. Vectorizable like asinf_fast. atan2f_fast(0, 0) is NaN where
// libm returns 0.
static inline float atan2f_fast(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y);
    // a = tan(angle folded to [0, pi/4]), in [0, 1]. Ternary min/max instead
    // of fminf/fmaxf: the libm calls have NaN semantics that don't map to
    // vmin/vmaxps, which blocks vectorization.
    const float mn = ax < ay ? ax : ay, mx = ax < ay ? ay : ax;
    const float a = mn / mx;
    // Cephes-style second reduction to |t| <= tan(pi/8)
    const bool red = a > 0.4142135623730950f;
    const float t = red ? (a - 1.0f) / (a + 1.0f) : a;
    const float z = t * t;
    float r = 8.05374449538e-2f;
    r = r * z - 1.38776856032e-1f;
    r = r * z + 1.99777106478e-1f;
    r = r * z - 3.33329491539e-1f;
    r = r * z * t + t;
    r = red ? r + 0.25f * kPI : r;
    // Undo the min/max fold and the quadrant fold
    r = ay > ax ? 0.5f * kPI - r : r;
    r = x < 0.0f ? kPI - r : r;
    return copysignf(r, y);
}

// Max cached 1D interpolation window (taps <= order + 1) for the lanczos and
// knab interpolators. Taps past the cap (order > 32) are dropped.
#define INTERP_MAX_TAPS 33

// Weighted sum of one contiguous row of samples. For complex input
// accumulates the real and imaginary parts in plain floats: gcc optimizes
// this much better than the c10::complex operator chain, especially with the
// hardening flags (-fstack-protector-strong) most Python builds add.
template<class T>
static inline T interp_row_cpu(const T *row, const float *w, int count) {
    if constexpr (std::is_same_v<T, complex64_t>) {
        const float *rowf = (const float*)row;
        float sr = 0.0f, si = 0.0f;
        for (int j = 0; j < count; j++) {
            sr += rowf[2*j] * w[j];
            si += rowf[2*j+1] * w[j];
        }
        return {sr, si};
    } else {
        T sum{};
        for (int j = 0; j < count; j++) {
            sum += row[j] * w[j];
        }
        return sum;
    }
}

// 1D windowed-sinc resampler tap. Reads the input signal at continuous
// position ``src`` (in input samples) with a Lanczos kernel. ``cutoff`` <= 1
// lowpasses to ``cutoff`` * input-Nyquist for anti-aliased decimation; it is 1
// for up/equal-rate sampling, giving the plain Lanczos kernel. Mirrors
// lanczos_resample_1d in cuda/util.h.
template<class T>
static T lanczos_resample_1d_cpu(const T *img, int n, float src, int order, float cutoff) {
    float a = 0.5f * order;
    float half = a / cutoff;
    int start = std::max(0, (int)ceilf(src - half));
    int end = std::min(n-1, (int)floorf(src + half));
    int count = std::min(end - start + 1, INTERP_MAX_TAPS);
    float w[INTERP_MAX_TAPS];
    for (int j = 0; j < count; j++) {
        w[j] = cutoff * lanczos_kernel(cutoff * (src - (start + j)), a);
    }
    return interp_row_cpu<T>(img + start, w, count);
}

template<class T>
static T lanczos_interp_2d_cpu(const T *img, int nx, int ny, float x, float y, int order) {
    float a = 0.5f * order;
    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    int start_y = std::max(0, (int)ceilf(y - a));
    int end_y = std::min(ny-1, (int)floorf(y + a));
    // Cache the y weights, they are reused for every x row.
    int ny_count = std::min(end_y - start_y + 1, INTERP_MAX_TAPS);
    float wy[INTERP_MAX_TAPS];
    for (int j = 0; j < ny_count; j++) {
        wy[j] = lanczos_kernel(y - (start_y + j), a);
    }
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = lanczos_kernel(dx, a);
        T row_val = interp_row_cpu<T>(img + i * ny + start_y, wy, ny_count);
        sum += row_val * wx;
    }
    return sum;
}

// Build a polyphase interpolation table: nphase+1 phases x taps weights,
// laid out phase-major so a hot loop gathers one contiguous tap row.
// `kernel(x)` is evaluated in double at the tap offset x of each phase;
// it is called once per table entry at setup time, so passing it as a
// template functor keeps the call inlined and off the hot path.
template<class KernelFn>
static std::vector<float> build_polyphase_table(int taps, int nphase,
                                                KernelFn kernel) {
    std::vector<float> table((size_t)(nphase + 1) * taps);
    for (int p = 0; p <= nphase; p++) {
        const double frac = (double)p / nphase;
        for (int j = 0; j < taps; j++) {
            const double x = frac + (taps / 2 - 1) - j;
            table[(size_t)p * taps + j] = (float)kernel(x);
        }
    }
    return table;
}

// Knab windowed sinc in double, as the exp form used by knab_kernel:
// with norm = exp(-2*pi*v*a) it is exactly sinc(x)*cosh(A*s)/cosh(A) for
// A = pi*v*a and s = sqrt(1 - (x/a)^2), but without the cosh overflow.
static inline double knab_kernel_double(double x, double a, double v) {
    if (std::fabs(x) >= a) {
        return 0.0;
    }
    if (x == 0.0) {
        return 1.0;
    }
    const double xa = x / a;
    const double sinc = std::sin(M_PI * x) / (M_PI * x);
    if (v <= 0.0) {
        return sinc;
    }
    const double norm = std::exp(-2.0 * a * M_PI * v);
    const double n = std::exp(M_PI * a * v * (std::sqrt(1.0 - xa * xa) - 1.0));
    return sinc * (norm / (n * (norm + 1.0)) + n / (norm + 1.0));
}

// 1D windowed-sinc resampler tap using the Knab kernel. See
// lanczos_resample_1d_cpu for the ``cutoff`` semantics. Mirrors
// knab_resample_1d in cuda/util.h.
template<class T>
static T knab_resample_1d_cpu(const T *img, int n, float src, int order, float v, float norm, float cutoff) {
    float a = 0.5f * order;
    float half = a / cutoff;
    int start = std::max(0, (int)ceilf(src - half));
    int end = std::min(n-1, (int)floorf(src + half));
    int count = std::min(end - start + 1, INTERP_MAX_TAPS);
    float w[INTERP_MAX_TAPS];
    for (int j = 0; j < count; j++) {
        w[j] = cutoff * knab_kernel(cutoff * (src - (start + j)), a, v, norm);
    }
    return interp_row_cpu<T>(img + start, w, count);
}

template<class T>
static T knab_interp_2d_cpu(const T *img, int nx, int ny, float x, float y, int order, float v, float norm) {
    float a = 0.5f * order;
    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    int start_y = std::max(0, (int)ceilf(y - a));
    int end_y = std::min(ny-1, (int)floorf(y + a));
    // Cache the y weights, they are reused for every x row.
    int ny_count = std::min(end_y - start_y + 1, INTERP_MAX_TAPS);
    float wy[INTERP_MAX_TAPS];
    for (int j = 0; j < ny_count; j++) {
        wy[j] = knab_kernel(y - (start_y + j), a, v, norm);
    }
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel(dx, a, v, norm);
        T row_val = interp_row_cpu<T>(img + i * ny + start_y, wy, ny_count);
        sum += row_val * wx;
    }
    return sum;
}

// knab_interp_2d_cpu with the x-axis (range) taps demodulated by ``fmod_x``
// (radians per x-sample): interpolates s(n)*exp(-j*fmod_x*(n - x)), which
// reconstructs s(x) exactly when the signal's local spectrum is centered at
// fmod_x instead of DC. Mirrors knab_interp_2d_fmod in cuda/util.h.
template<class T>
static T knab_interp_2d_fmod_cpu(const T *img, int nx, int ny, float x, float y,
                                 int order, float v, float norm, float fmod_x) {
    if (fmod_x == 0.0f) {
        return knab_interp_2d_cpu<T>(img, nx, ny, x, y, order, v, norm);
    }
    float a = 0.5f * order;
    int start_x = std::max(0, (int)ceilf(x - a));
    int end_x = std::min(nx-1, (int)floorf(x + a));
    int start_y = std::max(0, (int)ceilf(y - a));
    int end_y = std::min(ny-1, (int)floorf(y + a));
    int ny_count = std::min(end_y - start_y + 1, INTERP_MAX_TAPS);
    float wy[INTERP_MAX_TAPS];
    for (int j = 0; j < ny_count; j++) {
        wy[j] = knab_kernel(y - (start_y + j), a, v, norm);
    }
    T rot(cosf(-fmod_x * (start_x - x)), sinf(-fmod_x * (start_x - x)));
    const T step(cosf(-fmod_x), sinf(-fmod_x));
    T sum{};
    for (int i = start_x; i <= end_x; i++) {
        float dx = x - i;
        float wx = knab_kernel(dx, a, v, norm);
        T row_val = interp_row_cpu<T>(img + i * ny + start_y, wy, ny_count);
        sum += row_val * rot * wx;
        rot = rot * step;
    }
    return sum;
}

}
#endif
