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

#include "distance_matrix_accum_fp16.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

//! Calculate Fused-Multiply-Add (AVX512)
#define FMA_FP32_AVX512(zmm_m, zmm_q, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_q, zmm_sum);
#define FMA_MASK_FP32_AVX512(zmm_m, zmm_q, zmm_sum, mask) \
  zmm_sum = _mm512_mask3_fmadd_ps(zmm_m, zmm_q, zmm_sum, mask);

#define HorizontalAdd_FP16_NEON(v) \
  vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(v)), vcvt_high_f32_f16(v)))

#define HorizontalAdd_FP32_V512_TO_V256(zmm) \
  _mm256_add_ps(                             \
      _mm512_castps512_ps256(zmm),           \
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm), 1)))

//! Calculate Fused-Multiply-Add (AVX, FP16)
#define FMA_FP16_GENERAL(lhs, rhs, sum, norm1, norm2) \
  {                                                   \
    float v1 = lhs;                                   \
    float v2 = rhs;                                   \
    sum += v1 * v2;                                   \
    norm1 += v1 * v1;                                 \
    norm2 += v2 * v2;                                 \
  }

#if defined(__ARM_NEON) && defined(__aarch64__)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormNEON(const Float16 *lhs,
                                                   const Float16 *rhs,
                                                   size_t size, float *sql,
                                                   float *sqr) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 3) << 3);
  float16x8_t v_sum = vdupq_n_f16(0);
  float16x8_t v_sum_norm1 = vdupq_n_f16(0);
  float16x8_t v_sum_norm2 = vdupq_n_f16(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float16x8_t v_lhs = vld1q_f16((const float16_t *)lhs);
    float16x8_t v_rhs = vld1q_f16((const float16_t *)rhs);
    v_sum = vfmaq_f16(v_sum, v_lhs, v_rhs);
    v_sum_norm1 = vfmaq_f16(v_sum_norm1, v_lhs, v_lhs);
    v_sum_norm2 = vfmaq_f16(v_sum_norm2, v_rhs, v_rhs);
  }
  if (last >= last_aligned + 4) {
    float16x8_t v_lhs = vcombine_f16(vld1_f16((const float16_t *)lhs),
                                     vreinterpret_f16_u64(vdup_n_u64(0ul)));
    float16x8_t v_rhs = vcombine_f16(vld1_f16((const float16_t *)rhs),
                                     vreinterpret_f16_u64(vdup_n_u64(0ul)));
    v_sum = vfmaq_f16(v_sum, v_lhs, v_rhs);
    v_sum_norm1 = vfmaq_f16(v_sum_norm1, v_lhs, v_lhs);
    v_sum_norm2 = vfmaq_f16(v_sum_norm2, v_rhs, v_rhs);
    lhs += 4;
    rhs += 4;
  }

  float result = HorizontalAdd_FP16_NEON(v_sum);
  float norm1 = HorizontalAdd_FP16_NEON(v_sum_norm1);
  float norm2 = HorizontalAdd_FP16_NEON(v_sum_norm2);

  switch (last - lhs) {
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}
#else
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormNEON(const Float16 *lhs,
                                                   const Float16 *rhs,
                                                   size_t size, float *sql,
                                                   float *sqr) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 3) << 3);
  float32x4_t v_sum_0 = vdupq_n_f32(0);
  float32x4_t v_sum_1 = vdupq_n_f32(0);
  float32x4_t v_sum_norm1 = vdupq_n_f32(0);
  float32x4_t v_sum_norm2 = vdupq_n_f32(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float16x8_t v_lhs = vld1q_f16((const float16_t *)lhs);
    float16x8_t v_rhs = vld1q_f16((const float16_t *)rhs);
    float32x4_t v_lhs_0 = vcvt_f32_f16(vget_low_f16(v_lhs));
    float32x4_t v_rhs_0 = vcvt_f32_f16(vget_low_f16(v_rhs));
    float32x4_t v_lhs_1 = vcvt_high_f32_f16(v_lhs);
    float32x4_t v_rhs_1 = vcvt_high_f32_f16(v_rhs);
    v_sum_0 = vfmaq_f32(v_sum_0, v_lhs_0, v_rhs_0);
    v_sum_1 = vfmaq_f32(v_sum_1, v_lhs_1, v_rhs_1);
    v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_0, v_lhs_0);
    v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_1, v_lhs_1);
    v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_0, v_rhs_0);
    v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_1, v_rhs_1);
  }
  if (last >= last_aligned + 4) {
    float32x4_t v_lhs_0 = vcvt_f32_f16(vld1_f16((const float16_t *)lhs));
    float32x4_t v_rhs_0 = vcvt_f32_f16(vld1_f16((const float16_t *)rhs));
    v_sum_0 = vfmaq_f32(v_sum_0, v_lhs_0, v_rhs_0);
    v_sum_norm1 = vfmaq_f32(v_sum_norm1, v_lhs_0, v_lhs_0);
    v_sum_norm2 = vfmaq_f32(v_sum_norm2, v_rhs_0, v_rhs_0);
    lhs += 4;
    rhs += 4;
  }

  float result = vaddvq_f32(vaddq_f32(v_sum_0, v_sum_1));
  float norm1 = vaddvq_f32(v_sum_norm1);
  float norm2 = vaddvq_f32(v_sum_norm2);
  switch (last - lhs) {
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif  // __ARM_NEON && __aarch64__

#if defined(__AVX__) && defined(__F16C__)
#if defined(__AVX512F__)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormAVX512(const Float16 *lhs,
                                                     const Float16 *rhs,
                                                     size_t size, float *sql,
                                                     float *sqr) {
  __m512 zmm_sum_0 = _mm512_setzero_ps();
  __m512 zmm_sum_1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm1 = _mm512_setzero_ps();
  __m512 zmm_sum_norm2 = _mm512_setzero_ps();

  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 5) << 5);
  if (((uintptr_t)lhs & 0x3f) == 0 && ((uintptr_t)rhs & 0x3f) == 0) {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512i zmm_lhs = _mm512_load_si512((const __m512i *)lhs);
      __m512i zmm_rhs = _mm512_load_si512((const __m512i *)rhs);
      __m512 zmm_lhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_lhs));
      __m512 zmm_lhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_lhs, 1));
      __m512 zmm_rhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_rhs));
      __m512 zmm_rhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_rhs, 1));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }
    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 =
          _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)lhs));
      __m512 zmm_rhs_0 =
          _mm512_cvtph_ps(_mm256_load_si256((const __m256i *)rhs));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 32, rhs += 32) {
      __m512i zmm_lhs = _mm512_loadu_si512((const __m512i *)lhs);
      __m512i zmm_rhs = _mm512_loadu_si512((const __m512i *)rhs);
      __m512 zmm_lhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_lhs));
      __m512 zmm_lhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_lhs, 1));
      __m512 zmm_rhs_0 = _mm512_cvtph_ps(_mm512_castsi512_si256(zmm_rhs));
      __m512 zmm_rhs_1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(zmm_rhs, 1));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_rhs_1, zmm_sum_1)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_lhs_1, zmm_lhs_1, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      FMA_FP32_AVX512(zmm_rhs_1, zmm_rhs_1, zmm_sum_norm2)
    }
    if (last >= last_aligned + 16) {
      __m512 zmm_lhs_0 =
          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)lhs));
      __m512 zmm_rhs_0 =
          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)rhs));
      FMA_FP32_AVX512(zmm_lhs_0, zmm_rhs_0, zmm_sum_0)
      FMA_FP32_AVX512(zmm_lhs_0, zmm_lhs_0, zmm_sum_norm1)
      FMA_FP32_AVX512(zmm_rhs_0, zmm_rhs_0, zmm_sum_norm2)
      lhs += 16;
      rhs += 16;
    }
  }

  __m256 ymm_sum_0 =
      HorizontalAdd_FP32_V512_TO_V256(_mm512_add_ps(zmm_sum_0, zmm_sum_1));
  __m256 ymm_sum_norm1 = HorizontalAdd_FP32_V512_TO_V256(zmm_sum_norm1);
  __m256 ymm_sum_norm2 = HorizontalAdd_FP32_V512_TO_V256(zmm_sum_norm2);
  if (last >= lhs + 8) {
    __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)lhs));
    __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)rhs));
    ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
    ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
    ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
    lhs += 8;
    rhs += 8;
  }

  float result = HorizontalAdd_FP32_V256(ymm_sum_0);
  float norm1 = HorizontalAdd_FP32_V256(ymm_sum_norm1);
  float norm2 = HorizontalAdd_FP32_V256(ymm_sum_norm2);
  switch (last - lhs) {
    case 7:
      FMA_FP16_GENERAL(lhs[6], rhs[6], result, norm1, norm2);
      /* FALLTHRU */
    case 6:
      FMA_FP16_GENERAL(lhs[5], rhs[5], result, norm1, norm2);
      /* FALLTHRU */
    case 5:
      FMA_FP16_GENERAL(lhs[4], rhs[4], result, norm1, norm2);
      /* FALLTHRU */
    case 4:
      FMA_FP16_GENERAL(lhs[3], rhs[3], result, norm1, norm2);
      /* FALLTHRU */
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }

  *sql = norm1;
  *sqr = norm2;
  return result;
}
#else
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static inline float InnerProductAndSquaredNormAVX(const Float16 *lhs,
                                                  const Float16 *rhs,
                                                  size_t size, float *sql,
                                                  float *sqr) {
  __m256 ymm_sum_0 = _mm256_setzero_ps();
  __m256 ymm_sum_1 = _mm256_setzero_ps();
  __m256 ymm_sum_norm1 = _mm256_setzero_ps();
  __m256 ymm_sum_norm2 = _mm256_setzero_ps();

  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 4) << 4);
  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_lhs));
      __m256 ymm_lhs_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_lhs, 1));
      __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_rhs));
      __m256 ymm_rhs_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_rhs, 1));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }
    if (last >= last_aligned + 8) {
      __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)lhs));
      __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i *)rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != last_aligned; lhs += 16, rhs += 16) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_lhs));
      __m256 ymm_lhs_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_lhs, 1));
      __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm256_castsi256_si128(ymm_rhs));
      __m256 ymm_rhs_1 = _mm256_cvtph_ps(_mm256_extractf128_si256(ymm_rhs, 1));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_rhs_1, ymm_sum_1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_1, ymm_lhs_1, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_1, ymm_rhs_1, ymm_sum_norm2);
    }
    if (last >= last_aligned + 8) {
      __m256 ymm_lhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)lhs));
      __m256 ymm_rhs_0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)rhs));
      ymm_sum_0 = _mm256_fmadd_ps(ymm_lhs_0, ymm_rhs_0, ymm_sum_0);
      ymm_sum_norm1 = _mm256_fmadd_ps(ymm_lhs_0, ymm_lhs_0, ymm_sum_norm1);
      ymm_sum_norm2 = _mm256_fmadd_ps(ymm_rhs_0, ymm_rhs_0, ymm_sum_norm2);
      lhs += 8;
      rhs += 8;
    }
  }

  float result = HorizontalAdd_FP32_V256(_mm256_add_ps(ymm_sum_0, ymm_sum_1));
  float norm1 = HorizontalAdd_FP32_V256(ymm_sum_norm1);
  float norm2 = HorizontalAdd_FP32_V256(ymm_sum_norm2);
  switch (last - lhs) {
    case 7:
      FMA_FP16_GENERAL(lhs[6], rhs[6], result, norm1, norm2);
      /* FALLTHRU */
    case 6:
      FMA_FP16_GENERAL(lhs[5], rhs[5], result, norm1, norm2);
      /* FALLTHRU */
    case 5:
      FMA_FP16_GENERAL(lhs[4], rhs[4], result, norm1, norm2);
      /* FALLTHRU */
    case 4:
      FMA_FP16_GENERAL(lhs[3], rhs[3], result, norm1, norm2);
      /* FALLTHRU */
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }

  *sql = norm1;
  *sqr = norm2;
  return result;
}
#endif  // __AVX512F__
#endif  // __AVX__ && __F16C__

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the distance between matrix and query by SphericalInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, float e2, float *out) {
  float u2;
  float v2;
  float sum;

#if defined(__ARM_NEON)
  sum = InnerProductAndSquaredNormNEON(p, q, dim, &u2, &v2);
#elif defined(__AVX512F__)
  sum = InnerProductAndSquaredNormAVX512(p, q, dim, &u2, &v2);
#else
  sum = InnerProductAndSquaredNormAVX(p, q, dim, &u2, &v2);
#endif

  *out = ComputeSphericalInjection(sum, u2, v2, e2);
}

//! Compute the distance between matrix and query by RepeatedQuadraticInjection
void MipsSquaredEuclideanDistanceMatrix<Float16, 1, 1>::Compute(
    const ValueType *p, const ValueType *q, size_t dim, size_t m, float e2,
    float *out) {
  float u2;
  float v2;
  float sum;

#if defined(__ARM_NEON)
  sum = InnerProductAndSquaredNormNEON(p, q, dim, &u2, &v2);
#elif defined(__AVX512F__)
  sum = InnerProductAndSquaredNormAVX512(p, q, dim, &u2, &v2);
#else
  sum = InnerProductAndSquaredNormAVX(p, q, dim, &u2, &v2);
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
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec