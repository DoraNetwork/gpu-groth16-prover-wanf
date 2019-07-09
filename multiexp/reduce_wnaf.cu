#include "reduce.cu"

template< typename T >
__device__ void dump(T *n, int start, int end) {
    if (threadIdx.x == 0) {
        for (int i = start; i < end; i++) {
            printf("dump [%d]=%x\n", i, n[i]);
        }
    }
}

// C is the size of the precomputation
// R is the number of points we're handling per thread
template< typename EC, int C = 4, int RR = 8 >
__global__ void
ec_multiexp_wnaf(var *out, const var *multiples_, const int8_t *wnaf_, size_t N)
{
    int T = threadIdx.x, B = blockIdx.x, D = blockDim.x;
    int elts_per_block = D / BIG_WIDTH;
    int tileIdx = T / BIG_WIDTH;

    int idx = elts_per_block * B + tileIdx;

    size_t n = (N + RR - 1) / RR;
    if (idx < n) {
        // TODO: Treat remainder separately so R can remain a compile time constant
        size_t R = (idx < n - 1) ? RR : (N % RR);

        typedef typename EC::group_type Fr;
        const size_t WNAF_SIZE = 769;
        static constexpr int JAC_POINT_LIMBS = 3 * EC::field_type::DEGREE * ELT_LIMBS;
        static constexpr int AFF_POINT_LIMBS = 2 * EC::field_type::DEGREE * ELT_LIMBS;
        int out_off = idx * JAC_POINT_LIMBS;
        int m_off = idx * RR * AFF_POINT_LIMBS;
        int w_off = idx * RR * WNAF_SIZE;

        const var *multiples = multiples_ + m_off;
        const int8_t *wnaf = wnaf_ + w_off;

        EC x;
        EC::set_zero(x);
        int i = WNAF_SIZE - 1;
        bool found_nonzero = false;
        while (i >= 0) {
            if (found_nonzero) {
                EC::dbl(x, x);
            }

            for (int j = 0; j < R; j++) {
                auto off = j * WNAF_SIZE + i;
                if (wnaf[off] != 0) {
                    found_nonzero = true;
                    EC m;
                    if (wnaf[off] > 0) {
                        EC::load_affine(m, multiples + ((wnaf[off]/2)*2*N + j)*AFF_POINT_LIMBS);
                    } else {
                        EC::load_affine(m, multiples + (((-wnaf[off]/2)*2+1)*N + j)*AFF_POINT_LIMBS);
                    }
                    EC::mixed_add(x, x, m);
                }
            }

            i--;
        }
        EC::store_jac(out + out_off, x);
    }
}

template< typename EC, int C, int R >
void
ec_reduce_wnaf(cudaStream_t &strm, var *out, const var *multiples, const int8_t *wnaf, size_t N)
{
    cudaStreamCreate(&strm);

    static constexpr size_t pt_limbs = EC::NELTS * ELT_LIMBS;
    size_t n = (N + R - 1) / R;

    size_t nblocks = (n * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

    ec_multiexp_wnaf<EC, C, R><<< nblocks, threads_per_block, 0, strm>>>(out, multiples, wnaf, N);

    size_t r = n & 1, m = n / 2;
    for ( ; m != 0; r = m & 1, m >>= 1) {
        nblocks = (m * BIG_WIDTH + threads_per_block - 1) / threads_per_block;

        ec_sum_all<EC><<<nblocks, threads_per_block, 0, strm>>>(out, out + m*pt_limbs, m);
        if (r)
            ec_sum_all<EC><<<1, threads_per_block, 0, strm>>>(out, out + 2*m*pt_limbs, 1);
    }
}

