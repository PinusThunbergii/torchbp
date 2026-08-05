#ifndef _TORCHBP_UTIL_SHARED_H
#define _TORCHBP_UTIL_SHARED_H
// Helpers shared verbatim between the CPU (cpu/util.h) and CUDA (cuda/util.h)
// backends: constants, bilinear interpolation, the lanczos/knab kernel
// functions and the tx_power helpers whose CPU and GPU results must stay
// numerically identical. Everything here compiles both as plain host C++ and
// as CUDA device code. TBP_HOST_DEVICE marks the shared functions and the
// tbp_ldg/tbp_sinpif shims paper over the few intrinsic differences.
#include <cmath>
#include <type_traits>

#if defined(__CUDACC__)
#define TBP_HOST_DEVICE __host__ __device__
#else
#define TBP_HOST_DEVICE
#endif

#define kPI 3.1415926535897932384626433f
#define kC0 299792458.0f

namespace torchbp {

// Read-only cached load on CUDA device code, plain load on host.
template<class T>
TBP_HOST_DEVICE static inline T tbp_ldg(const T* p) {
#if defined(__CUDA_ARCH__)
    return __ldg(p);
#else
    return *p;
#endif
}

// Host (sin(pi*x), cos(pi*x)). Device code uses the sincospif intrinsic
// instead.
template <typename T>
static void sincospi(T x, T *sinx, T *cosx) {
    if constexpr (std::is_same_v<T, float>) {
        // Fast float (sin(pi*x), cos(pi*x)). Working in "pi units" makes range
        // reduction exact and cheap: the period in x is 2, so a single rintf
        // + subtract maps x to r in [-0.5, 0.5].
        // CPU analogue of CUDA's __sincosf.
        const float k = rintf(x); // nearest integer, ties to even
        const float r = x - k;    // r in [-0.5, 0.5]
        const float u = r * r;
        // sinpi(r) = r * Ps(r^2)
        float s = 0.0778885794857989f;
        s = s * u - 0.5983952894635156f;
        s = s * u + 2.550091904145188f;
        s = s * u - 5.167710704395367f;
        s = s * u + 3.1415926441706428f;
        s *= r;
        // cospi(r) = Pc(r^2)
        float c = 0.22049100664642599f;
        c = c * u - 1.3322375115287677f;
        c = c * u + 4.058461261992171f;
        c = c * u - 4.934794985432785f;
        c = c * u + 0.9999999672684953f;
        // Odd integer part of x flips the sign of both sinpi and cospi.
        // Branchless so the function stays vectorizable when inlined into
        // SIMD loops (and the branch is data-dependent, i.e. unpredictable).
        const float sgn = 1.0f - 2.0f * (float)(((int)k) & 1);
        *sinx = s * sgn;
        *cosx = c * sgn;
    } else {
        *sinx = sin(static_cast<T>(kPI) * x);
        *cosx = cos(static_cast<T>(kPI) * x);
    }
}

// sin(pi*x): sinpif intrinsic on CUDA, the sincospi polynomial on host.
TBP_HOST_DEVICE static inline float tbp_sinpif(float x) {
#if defined(__CUDA_ARCH__)
    return sinpif(x);
#else
    float s, c;
    sincospi<float>(x, &s, &c);
    return s;
#endif
}

template<class T>
TBP_HOST_DEVICE static T interp2d(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return img[x_int*ny + y_int]*(1.0f-x_frac)*(1.0f-y_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac)*y_frac +
           img[(x_int+1)*ny + y_int]*x_frac*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*x_frac*y_frac;
}

template<class T>
TBP_HOST_DEVICE static T interp2d_gradx(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-y_frac) +
           -img[x_int*ny + y_int+1]*y_frac +
           img[(x_int+1)*ny + y_int]*(1.0f-y_frac) +
           img[(x_int+1)*ny + y_int+1]*y_frac;
}

template<class T>
TBP_HOST_DEVICE static T interp2d_grady(const T *img, int nx, int ny,
        int x_int, float x_frac, int y_int, float y_frac) {
    return -img[x_int*ny + y_int]*(1.0f-x_frac) +
           img[x_int*ny + y_int+1]*(1.0f-x_frac) +
           -img[(x_int+1)*ny + y_int]*x_frac +
           img[(x_int+1)*ny + y_int+1]*x_frac;
}

TBP_HOST_DEVICE static inline float lanczos_kernel(float x, float a) {
    // |x| >= a is ensured by calling code
    if (x == 0.0f) {
        return 1.0f;
    }
    return tbp_sinpif(x) / (kPI * x) * tbp_sinpif(x/a) / (kPI * x / a);
}

TBP_HOST_DEVICE static inline float knab_kernel_norm(int order, float v) {
    float a = 0.5f * order;
    return expf(-2.0f*a*kPI*v);
}

TBP_HOST_DEVICE static inline float knab_kernel(float x, float a, float v, float norm) {
    // This is needed due to rounding errors.
    if (fabsf(x) >= a) {
        return 0.0f;
    }
    if (x == 0.0f) {
        return 1.0f;
    }
    float xa = x / a;
    float n = expf(kPI * a * v * (sqrtf(1.0f - xa*xa) - 1.0f));
    return (tbp_sinpif(x) / (kPI * x)) * (norm/(n*(norm + 1.0f)) + n/(norm + 1.0f));
}

// Shared tx_power helpers.
//
// Edge-clamped bilinear sample of the 3 tx_power DEM channels (z, dz/dx,
// dz/dy) at fractional cell (fr, ft). Same cell-start index convention as the
// image->DEM mapping fr = idr * dem_nr / nr used by the coherent kernels.
TBP_HOST_DEVICE static inline void tx_power_dem_sample(const float* dem, int dem_nr,
        int dem_ntheta, float fr, float ft,
        float* z, float* dzdx, float* dzdy) {
    int ir0 = (int)fr;
    ir0 = ir0 < dem_nr - 1 ? ir0 : dem_nr - 1;
    const int ir1 = ir0 + 1 < dem_nr ? ir0 + 1 : dem_nr - 1;
    const float wr = fr - ir0;
    int it0 = (int)ft;
    it0 = it0 < dem_ntheta - 1 ? it0 : dem_ntheta - 1;
    const int it1 = it0 + 1 < dem_ntheta ? it0 + 1 : dem_ntheta - 1;
    const float wt = ft - it0;
    const size_t np = (size_t)dem_nr * dem_ntheta;
    float out[3];
    for (int c = 0; c < 3; c++) {
        const float* row0 = dem + c * np + (size_t)ir0 * dem_ntheta;
        const float* row1 = dem + c * np + (size_t)ir1 * dem_ntheta;
        const float a = tbp_ldg(&row0[it0])
                + wt * (tbp_ldg(&row0[it1]) - tbp_ldg(&row0[it0]));
        const float b = tbp_ldg(&row1[it0])
                + wt * (tbp_ldg(&row1[it1]) - tbp_ldg(&row1[it0]));
        out[c] = a + wr * (b - a);
    }
    *z = out[0]; *dzdx = out[1]; *dzdy = out[2];
}

// Accumulate the transmit-power moments for one ground pixel over all sweeps.
// px_base, py_base: pixel ground position. Per sweep the platform height is
// h_fixed (slant grid) or the sweep pos z (ground grid). min_sin2 is the floor
// on sin^2 of the look angle for the sigma/gamma normalizations. Fills
// pixel = sum wi/sinl and the Welford weighted moments (m_w, m_mean, m_s) of
// the ground-frame line-of-sight azimuth. Shared by the direct and factorized
// (accum) polar and Cartesian tx_power kernels.
//
// With HasDem the pixel sits at height z_base with Cartesian terrain slopes
// (dzdx, dzdy): the platform height over the pixel becomes pos_z - z_base
// (also fixing the antenna elevation lookup), and sigma/gamma use the local
// patch orientation instead of the flat-ground incidence. With up-range slope
// u = s/Rg along the ground LOS and normal magnitude N = sqrt(1+|grad z|^2):
//   sigma_0: projected-area factor (Ulander) (sin(inc) - u*cos(inc))/N;
//            the azimuth slope enters only through N.
//   gamma_0: additionally divided by cos(local incidence) = (s+h)/(d*N)
//            (terrain-flattened gamma).
// Both clamp at the flat-formula floor (max(sqrt(f), x) == sqrt(max(f, x^2))
// for x >= 0, so zero slopes reproduce the flat expressions exactly). Layover
// folds clamp at the floor since tx_power is a divisor; shadowed patches
// (gamma) roll continuously to near-zero contribution via the clamped
// cos(local incidence).
template<bool HasDem = false>
TBP_HOST_DEVICE static inline void tx_power_pixel_moments(
        float px_base, float py_base, bool use_h_fixed, float h_fixed,
        float z_base, float dzdx, float dzdy,
        const float* pos, const float* att, int nsweeps,
        const float* g, float g_az0, float g_el0, float g_daz, float g_del,
        int g_naz, int g_nel, const float* wa, int normalization,
        float min_sin2,
        float* pixel, float* m_w, float* m_mean, float* m_s) {
    float acc = 0.0f, mw = 0.0f, mmean = 0.0f, ms = 0.0f;
    float inv_N = 1.0f, sin_floor = 0.0f;
    if constexpr (HasDem) {
        inv_N = 1.0f / sqrtf(1.0f + dzdx*dzdx + dzdy*dzdy);
        sin_floor = sqrtf(min_sin2);
    }
    for (int i = 0; i < nsweeps; i++) {
        const float px = px_base - pos[i*3 + 0];
        const float py = py_base - pos[i*3 + 1];
        float h = use_h_fixed ? h_fixed : pos[i*3 + 2];
        if constexpr (HasDem) h -= z_base;
        const float d = sqrtf(px*px + py*py + h*h);
        const float look_angle = asinf(fmaxf(-h / d, -1.0f));
        const float psi = atan2f(py, px);  // ground-frame LOS azimuth
        float el_a = look_angle - att[3*i + 0];
        float az_a = psi - att[3*i + 2];
        const float pitch = att[3*i + 1];
        if (pitch != 0.0f) {
            // Pitch rotates the antenna about its boresight (the along-track
            // attitude angle for a side-looking antenna). Rotate the
            // roll/yaw-compensated LOS about the pattern x axis with the full
            // spherical rotation (x = cos(el)cos(az), y = cos(el)sin(az),
            // z = sin(el)), matching a pattern rotated by the same angle
            // about [1, 0, 0]. Zero pitch takes the exact legacy path.
            const float ce = cosf(el_a);
            const float ux = ce * cosf(az_a);
            const float uy = ce * sinf(az_a);
            const float uz = sinf(el_a);
            const float cp = cosf(pitch);
            const float sp = sinf(pitch);
            const float uyp = cp * uy - sp * uz;
            const float uzp = sp * uy + cp * uz;
            el_a = asinf(fmaxf(-1.0f, fminf(1.0f, uzp)));
            az_a = atan2f(uyp, ux);
        }
        const float el_idx = (el_a - g_el0) / g_del;
        const float az_idx = (az_a - g_az0) / g_daz;
        const int el_int = el_idx;
        const int az_int = az_idx;
        // Reject samples below the pattern's first row/column (negative
        // fractional index) rather than extrapolating gain below the edge.
        if (el_idx < 0.0f || el_int + 1 >= g_nel) continue;
        if (az_idx < 0.0f || az_int + 1 >= g_naz) continue;
        const float g_i = interp2d<float>(g, g_nel, g_naz,
                el_int, el_idx - el_int, az_int, az_idx - az_int);
        float sinl = 1.0f;
        if constexpr (HasDem) {
            if (normalization == 1 || normalization == 2) {
                const float Rg2 = px*px + py*py;
                const float Rg = fmaxf(sqrtf(Rg2), 1e-6f);
                const float s = px*dzdx + py*dzdy;
                if (normalization == 1) {           // sigma_0
                    sinl = fmaxf(sin_floor, (Rg2 - s*h) * inv_N / (Rg * d));
                } else {                            // gamma_0
                    // cos(local incidence) = (s + h) / (d * N) goes to zero
                    // at grazing and negative in shadow. Clamp it at a small
                    // positive value so the shadowed contribution rolls
                    // continuously to (nearly) zero instead of jumping; the
                    // discontinuity would break the factorized (ffbp)
                    // interpolation at the shadow boundary.
                    const float floor_g = sin_floor * d / fmaxf(h, 1e-3f);
                    const float den = Rg * fmaxf(s + h, 1e-3f * d);
                    sinl = fmaxf(floor_g, (Rg2 - s*h) / den);
                }
            } else if (normalization == 3) {        // point (d^4)
                sinl = d;
            }
        } else {
            if (normalization == 1) {           // sigma_0
                sinl = sqrtf(fmaxf(min_sin2, 1.0f - (h*h)/(d*d)));
            } else if (normalization == 2) {    // gamma_0
                sinl = sqrtf(fmaxf(min_sin2, 1.0f - (h*h)/(d*d))) * d / h;
            } else if (normalization == 3) {    // point (d^4)
                sinl = d;
            }
        }
        const float w = wa[i];
        const float wi = g_i * g_i * w * w / (d*d*d);
        acc += wi / sinl;
        // Welford weighted update. Skip zero weights: with m_w still zero they
        // give 0/0 = NaN and poison the moments.
        if (wi > 0.0f) {
            const float wsum = mw + wi;
            const float delta = psi - mmean;
            mmean += delta * wi / wsum;
            ms += wi * delta * (psi - mmean);
            mw = wsum;
        }
    }
    *pixel = acc; *m_w = mw; *m_mean = mmean; *m_s = ms;
}

// Interpolate the 4 tx_power accumulator channels (S, W, P1, M2) of one input
// map at a fractional cell and combine them into the running merge totals with
// Chan's parallel-variance formula. Inputs with non-positive weight contribute
// nothing. Shared by the polar (ffbp_tx_power_merge2) and Cartesian
// (cart_tx_power_merge2) merge kernels.
TBP_HOST_DEVICE static inline void tx_power_merge_sample(const float* acc, int nx, int ny,
        int xi, float xf, int yi, float yf,
        float* S, float* W, float* P1, float* M2) {
    const size_t np = (size_t)nx * ny;
    const float w = interp2d<float>(&acc[1*np], nx, ny, xi, xf, yi, yf);
    if (w <= 0.0f) return;
    const float s  = interp2d<float>(&acc[0*np], nx, ny, xi, xf, yi, yf);
    const float p1 = interp2d<float>(&acc[2*np], nx, ny, xi, xf, yi, yf);
    const float m2 = interp2d<float>(&acc[3*np], nx, ny, xi, xf, yi, yf);
    if (*W > 0.0f) {
        // Chan's parallel variance combination of the weighted psi moments.
        const float delta = *P1 / *W - p1 / w;
        *M2 += m2 + delta * delta * (*W) * w / (*W + w);
    } else {
        *M2 += m2;
    }
    *S += s; *W += w; *P1 += p1;
}

// Element-wise a (*|/) interp2d(b) at one output element of the
// div/mul_2d_interp_linear ops: maps output cell (id0, id1) of the [Na0, Na1]
// grid onto the [Nb0, Nb1] input grid with edge-clamped bilinear
// interpolation. Shared by the CPU (OpenMP loop) and CUDA (kernel) callers.
template<typename T, typename T2>
TBP_HOST_DEVICE static inline void interp2d_combine_elem(const T* a, const T2* b, T* out,
        int Na0, int Na1, int Nb0, int Nb1, bool is_div, int idx, int idbatch) {
    const int id0 = idx / Na1;
    const int id1 = idx % Na1;

    if (id0 >= Na0 || id1 >= Na1) return;

    // Map from output grid [0, Na0-1] x [0, Na1-1] to input grid [0, Nb0-1] x [0, Nb1-1]
    float b0_float = (float)id0 * (Nb0 - 1) / (Na0 - 1);
    float b1_float = (float)id1 * (Nb1 - 1) / (Na1 - 1);

    int b0_int = (int)floorf(b0_float);
    int b1_int = (int)floorf(b1_float);
    float b0_frac = b0_float - b0_int;
    float b1_frac = b1_float - b1_int;

    // Clamp to valid range
    b0_int = b0_int < Nb0 - 2 ? b0_int : Nb0 - 2;
    b1_int = b1_int < Nb1 - 2 ? b1_int : Nb1 - 2;
    b0_int = b0_int > 0 ? b0_int : 0;
    b1_int = b1_int > 0 ? b1_int : 0;

    const T2 v = interp2d<T2>(&b[idbatch * Nb1 * Nb0], Nb0, Nb1, b0_int, b0_frac, b1_int, b1_frac);
    const int oidx = idbatch * Na1 * Na0 + id0 * Na1 + id1;
    if (is_div) {
        out[oidx] = a[oidx] / static_cast<T>(v);
    } else {
        out[oidx] = a[oidx] * static_cast<T>(v);
    }
}

}
#endif
