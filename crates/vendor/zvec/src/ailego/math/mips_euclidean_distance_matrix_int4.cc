// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "distance_matrix_accum_int8.i"
#include "inner_product_matrix.h"
#include "mips_euclidean_distance_matrix.h"
#include "norm_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__SSE4_1__)
//! Four-bits Convert Table
static const AILEGO_ALIGNED(32) int8_t Int4ConvertTable[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
#endif  // __SSE4_1__

#if defined(__SSE4_1__)
static const __m128i MASK_INT4_SSE = _mm_set1_epi32(0x0f0f0f0f);
static const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);
static const __m128i INT4_LOOKUP_SSE =
    _mm_load_si128((const __m128i *)Int4ConvertTable);
#endif  // __SSE4_1__

#if defined(__AVX2__)
static const __m256i MASK_INT4_AVX = _mm256_set1_epi32(0x0f0f0f0f);
static const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
static const __m256i INT4_LOOKUP_AVX =
    _mm256_load_si256((const __m256i *)Int4ConvertTable);
#endif  // __AVX2__

//! Calculate Fused-Multiply-Add (GENERAL)
#define FMA_INT4_GENERAL(lhs, rhs, sum, norm1, norm2)                   \
  {                                                                     \
    sum += Int4MulTable[(((lhs) << 4) & 0xf0) | (((rhs) >> 0) & 0xf)] + \
           Int4MulTable[(((lhs) >> 0) & 0xf0) | (((rhs) >> 4) & 0xf)];  \
    norm1 += static_cast<float>(                                        \
        ((int8_t)((lhs) << 4) >> 4) * ((int8_t)((lhs) << 4) >> 4) +     \
        ((int8_t)((lhs) & 0xf0) >> 4) * ((int8_t)((lhs) & 0xf0) >> 4)); \
    norm2 += static_cast<float>(                                        \
        ((int8_t)((rhs) << 4) >> 4) * ((int8_t)((rhs) << 4) >> 4) +     \
        ((int8_t)((rhs) & 0xf0) >> 4) * ((int8_t)((rhs) & 0xf0) >> 4)); \
  }

//! Calculate Fused-Multiply-Add (SSE)
#define FMA_INT8_SSE(xmm_lhs, xmm_rhs, xmm_sum)                          \
  xmm_sum = _mm_add_epi32(                                               \
      _mm_madd_epi16(_mm_maddubs_epi16(_mm_abs_epi8(xmm_rhs),            \
                                       _mm_sign_epi8(xmm_lhs, xmm_rhs)), \
                     ONES_INT16_SSE),                                    \
      xmm_sum)

//! Calculate Fused-Multiply-Add (AVX)
#define FMA_INT8_AVX(ymm_lhs, ymm_rhs, ymm_sum)                     \
  ymm_sum = _mm256_add_epi32(                                       \
      _mm256_madd_epi16(                                            \
          _mm256_maddubs_epi16(_mm256_abs_epi8(ymm_rhs),            \
                               _mm256_sign_epi8(ymm_lhs, ymm_rhs)), \
          ONES_INT16_AVX),                                          \
      ymm_sum)

//! Compute the distance between matrix and query (SSE)
#define FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum_0, xmm_sum_norm1, \
                          xmm_sum_norm2)                              \
  {                                                                   \
    __m128i xmm_lhs_0 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_lhs), MASK_INT4_SSE));    \
    __m128i xmm_rhs_0 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE, _mm_and_si128((xmm_rhs), MASK_INT4_SSE));    \
    __m128i xmm_lhs_1 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE,                                              \
        _mm_and_si128(_mm_srli_epi32((xmm_lhs), 4), MASK_INT4_SSE));  \
    __m128i xmm_rhs_1 = _mm_shuffle_epi8(                             \
        INT4_LOOKUP_SSE,                                              \
        _mm_and_si128(_mm_srli_epi32((xmm_rhs), 4), MASK_INT4_SSE));  \
    FMA_INT8_SSE(xmm_lhs_0, xmm_rhs_0, xmm_sum_0);                    \
    FMA_INT8_SSE(xmm_lhs_0, xmm_lhs_0, xmm_sum_norm1);                \
    FMA_INT8_SSE(xmm_rhs_0, xmm_rhs_0, xmm_sum_norm2);                \
    FMA_INT8_SSE(xmm_lhs_1, xmm_rhs_1, xmm_sum_0);                    \
    FMA_INT8_SSE(xmm_lhs_1, xmm_lhs_1, xmm_sum_norm1);                \
    FMA_INT8_SSE(xmm_rhs_1, xmm_rhs_1, xmm_sum_norm2);                \
  }

//! Compute the distance between matrix and query (AVX)
#define FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum_0, ymm_sum1,           \
                          ymm_sum_norm1, ymm_sum_norm2)                    \
  {                                                                        \
    __m256i ymm_lhs_0 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_lhs), MASK_INT4_AVX));      \
    __m256i ymm_rhs_0 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX, _mm256_and_si256((ymm_rhs), MASK_INT4_AVX));      \
    __m256i ymm_lhs_1 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX,                                                   \
        _mm256_and_si256(_mm256_srli_epi32((ymm_lhs), 4), MASK_INT4_AVX)); \
    __m256i ymm_rhs_1 = _mm256_shuffle_epi8(                               \
        INT4_LOOKUP_AVX,                                                   \
        _mm256_and_si256(_mm256_srli_epi32((ymm_rhs), 4), MASK_INT4_AVX)); \
    FMA_INT8_AVX(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);                         \
    FMA_INT8_AVX(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);                         \
    FMA_INT8_AVX(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);                     \
    FMA_INT8_AVX(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);                     \
    FMA_INT8_AVX(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);                     \
    FMA_INT8_AVX(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);                     \
  }

#if defined(__SSE4_1__)
#if defined(__AVX2__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormAVX(const uint8_t *lhs,
                                                  const uint8_t *rhs,
                                                  size_t size, float *sql,
                                                  float *sqr) {
  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 5) << 5);
  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();
  __m256i ymm_sum_norm1 = _mm256_setzero_si256();
  __m256i ymm_sum_norm2 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)(rhs));
      FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum_0, ymm_sum1, ymm_sum_norm1,
                        ymm_sum_norm2)
    }
    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      __m128i xmm_sum_norm1 = _mm_setzero_si128();
      __m128i xmm_sum_norm2 = _mm_setzero_si128();
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum, xmm_sum_norm1, xmm_sum_norm2)
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum), ymm_sum_0);
      ymm_sum_norm1 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum_norm1), ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum_norm2), ymm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)(lhs));
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)(rhs));
      FMA_INT4_ITER_AVX(ymm_lhs, ymm_rhs, ymm_sum_0, ymm_sum1, ymm_sum_norm1,
                        ymm_sum_norm2)
    }
    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      __m128i xmm_sum = _mm_setzero_si128();
      __m128i xmm_sum_norm1 = _mm_setzero_si128();
      __m128i xmm_sum_norm2 = _mm_setzero_si128();
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum, xmm_sum_norm1, xmm_sum_norm2)
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum), ymm_sum_0);
      ymm_sum_norm1 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum_norm1), ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(), xmm_sum_norm2), ymm_sum_norm2);
      lhs += 16;
      rhs += 16;
    }
  }
  float result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));
  float norm1 = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum_norm1));
  float norm2 = static_cast<float>(HorizontalAdd_INT32_V256(ymm_sum_norm2));

  switch (last - lhs) {
    case 15:
      FMA_INT4_GENERAL(lhs[14], rhs[14], result, norm1, norm2)
      /* FALLTHRU */
    case 14:
      FMA_INT4_GENERAL(lhs[13], rhs[13], result, norm1, norm2)
      /* FALLTHRU */
    case 13:
      FMA_INT4_GENERAL(lhs[12], rhs[12], result, norm1, norm2)
      /* FALLTHRU */
    case 12:
      FMA_INT4_GENERAL(lhs[11], rhs[11], result, norm1, norm2)
      /* FALLTHRU */
    case 11:
      FMA_INT4_GENERAL(lhs[10], rhs[10], result, norm1, norm2)
      /* FALLTHRU */
    case 10:
      FMA_INT4_GENERAL(lhs[9], rhs[9], result, norm1, norm2)
      /* FALLTHRU */
    case 9:
      FMA_INT4_GENERAL(lhs[8], rhs[8], result, norm1, norm2)
      /* FALLTHRU */
    case 8:
      FMA_INT4_GENERAL(lhs[7], rhs[7], result, norm1, norm2)
      /* FALLTHRU */
    case 7:
      FMA_INT4_GENERAL(lhs[6], rhs[6], result, norm1, norm2)
      /* FALLTHRU */
    case 6:
      FMA_INT4_GENERAL(lhs[5], rhs[5], result, norm1, norm2)
      /* FALLTHRU */
    case 5:
      FMA_INT4_GENERAL(lhs[4], rhs[4], result, norm1, norm2)
      /* FALLTHRU */
    case 4:
      FMA_INT4_GENERAL(lhs[3], rhs[3], result, norm1, norm2)
      /* FALLTHRU */
    case 3:
      FMA_INT4_GENERAL(lhs[2], rhs[2], result, norm1, norm2)
      /* FALLTHRU */
    case 2:
      FMA_INT4_GENERAL(lhs[1], rhs[1], result, norm1, norm2)
      /* FALLTHRU */
    case 1:
      FMA_INT4_GENERAL(lhs[0], rhs[0], result, norm1, norm2)
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}
#else
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormSSE(const uint8_t *lhs,
                                                  const uint8_t *rhs,
                                                  size_t size, float *sql,
                                                  float *sqr) {
  const uint8_t *last = lhs + size;
  const uint8_t *last_aligned = lhs + ((size >> 4) << 4);
  __m128i xmm_sum = _mm_setzero_si128();
  __m128i xmm_sum_norm1 = _mm_setzero_si128();
  __m128i xmm_sum_norm2 = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum, xmm_sum_norm1, xmm_sum_norm2)
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)(lhs));
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)(rhs));
      FMA_INT4_ITER_SSE(xmm_lhs, xmm_rhs, xmm_sum, xmm_sum_norm1, xmm_sum_norm2)
    }
  }
  float result = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum));
  float norm1 = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum_norm1));
  float norm2 = static_cast<float>(HorizontalAdd_INT32_V128(xmm_sum_norm2));

  switch (last - lhs) {
    case 15:
      FMA_INT4_GENERAL(lhs[14], rhs[14], result, norm1, norm2)
      /* FALLTHRU */
    case 14:
      FMA_INT4_GENERAL(lhs[13], rhs[13], result, norm1, norm2)
      /* FALLTHRU */
    case 13:
      FMA_INT4_GENERAL(lhs[12], rhs[12], result, norm1, norm2)
      /* FALLTHRU */
    case 12:
      FMA_INT4_GENERAL(lhs[11], rhs[11], result, norm1, norm2)
      /* FALLTHRU */
    case 11:
      FMA_INT4_GENERAL(lhs[10], rhs[10], result, norm1, norm2)
      /* FALLTHRU */
    case 10:
      FMA_INT4_GENERAL(lhs[9], rhs[9], result, norm1, norm2)
      /* FALLTHRU */
    case 9:
      FMA_INT4_GENERAL(lhs[8], rhs[8], result, norm1, norm2)
      /* FALLTHRU */
    case 8:
      FMA_INT4_GENERAL(lhs[7], rhs[7], result, norm1, norm2)
      /* FALLTHRU */
    case 7:
      FMA_INT4_GENERAL(lhs[6], rhs[6], result, norm1, norm2)
      /* FALLTHRU */
    case 6:
      FMA_INT4_GENERAL(lhs[5], rhs[5], result, norm1, norm2)
      /* FALLTHRU */
    case 5:
      FMA_INT4_GENERAL(lhs[4], rhs[4], result, norm1, norm2)
      /* FALLTHRU */
    case 4:
      FMA_INT4_GENERAL(lhs[3], rhs[3], result, norm1, norm2)
      /* FALLTHRU */
    case 3:
      FMA_INT4_GENERAL(lhs[2], rhs[2], result, norm1, norm2)
      /* FALLTHRU */
    case 2:
      FMA_INT4_GENERAL(lhs[1], rhs[1], result, norm1, norm2)
      /* FALLTHRU */
    case 1:
      FMA_INT4_GENERAL(lhs[0], rhs[0], result, norm1, norm2)
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}
#endif  // __AVX2__

//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
  float u2;
  float v2;
  float sum;

#if defined(__AVX2__)
  sum = InnerProductAndSquaredNormAVX(p, q, dim >> 1, &u2, &v2);
#else
  sum = InnerProductAndSquaredNormSSE(p, q, dim >> 1, &u2, &v2);
#endif

  *out = ComputeSphericalInjection(sum, u2, v2, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<uint8_t, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
  float u2;
  float v2;
  float sum;

#if defined(__AVX2__)
  sum = InnerProductAndSquaredNormAVX(p, q, dim >> 1, &u2, &v2);
#else
  sum = InnerProductAndSquaredNormSSE(p, q, dim >> 1, &u2, &v2);
#endif

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }
  *out = sum;
}
#endif  // __SSE4_1__

}  // namespace ailego
}  // namespace zvec