import logging
import math
import numpy as np

from zvec import (
    MetricType,
    DataType,
    QuantizeType,
    Doc,
    CollectionSchema,
    FieldSchema,
    VectorSchema,
)

from typing import Dict


def is_float_equal(actual, expected, rel_tol=1e-5, abs_tol=1e-8):
    if actual is None and expected is None:
        return True
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def is_dense_vector_equal(vec1, vec2, rtol=1e-5, atol=1e-8):
    """Compare two dense vectors with tolerance."""
    return np.allclose(vec1, vec2, rtol=rtol, atol=atol)


def is_sparse_vector_equal(vec1, vec2, rtol=1e-5, atol=1e-8):
    """Compare two sparse vectors with tolerance."""
    # Check if they have the same keys
    if set(vec1.keys()) != set(vec2.keys()):
        return False

    # Check if all values are close
    for key in vec1:
        if not math.isclose(vec1[key], vec2[key], rel_tol=rtol, abs_tol=atol):
            return False

    return True


def is_float_array_equal(arr1, arr2, rtol=1e-5, atol=1e-8):
    """Compare two float arrays with tolerance."""
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def is_double_array_equal(arr1, arr2, rtol=1e-9, atol=1e-12):
    """Compare two double arrays with tolerance."""
    return np.allclose(arr1, arr2, rtol=rtol, atol=atol)


def is_int_array_equal(arr1, arr2):
    """Compare two integer arrays with exact equality."""
    return np.array_equal(arr1, arr2)


def cosine_distance_dense(
    vec1,
    vec2,
    dtype: DataType = DataType.VECTOR_FP32,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    if dtype == DataType.VECTOR_FP16 or quantize_type == QuantizeType.FP16:
        vec1 = [np.float16(a) for a in vec1]
        vec2 = [np.float16(b) for b in vec2]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return 1 - dot_product / (magnitude1 * magnitude2)


def dp_distance_dense(
    vec1,
    vec2,
    dtype: DataType = DataType.VECTOR_FP32,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    if dtype == DataType.VECTOR_FP16 or quantize_type == QuantizeType.FP16:
        return sum(np.float16(a) * np.float16(b) for a, b in zip(vec1, vec2))
    return sum(a * b for a, b in zip(vec1, vec2))


def euclidean_distance_dense(
    vec1,
    vec2,
    dtype: DataType = DataType.VECTOR_FP32,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    if dtype == DataType.VECTOR_FP16 or quantize_type == QuantizeType.FP16:
        return sum((np.float16(a) - np.float16(b)) ** 2 for a, b in zip(vec1, vec2))
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2))


def distance_dense(
    vec1,
    vec2,
    metric: MetricType,
    data_type: DataType = DataType.VECTOR_FP32,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    if metric == MetricType.COSINE:
        return cosine_distance_dense(vec1, vec2, data_type, quantize_type)
    elif metric == MetricType.L2:
        return euclidean_distance_dense(vec1, vec2, data_type, quantize_type)
    elif metric == MetricType.IP:
        return dp_distance_dense(vec1, vec2, data_type, quantize_type)
    else:
        raise ValueError("Unsupported metric type")


def dp_distance_sparse(
    vec1,
    vec2,
    data_type: DataType = DataType.SPARSE_VECTOR_FP32,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    dot_product = 0.0
    for dim in set(vec1.keys()) & set(vec2.keys()):
        if (
            data_type == DataType.SPARSE_VECTOR_FP16
            or quantize_type == QuantizeType.FP16
        ):
            vec1[dim] = np.float16(vec1[dim])
            vec2[dim] = np.float16(vec2[dim])
        dot_product += vec1[dim] * vec2[dim]
    return dot_product


def distance(
    vec1,
    vec2,
    metric: MetricType,
    data_type: DataType,
    quantize_type: QuantizeType = QuantizeType.UNDEFINED,
):
    is_sparse = (
        data_type == DataType.SPARSE_VECTOR_FP32
        or data_type == DataType.SPARSE_VECTOR_FP16
    )

    if is_sparse:
        if metric != MetricType.IP:
            raise ValueError("Unsupported metric type for sparse vectors")

    if is_sparse:
        return dp_distance_sparse(vec1, vec2, data_type, quantize_type)
    else:
        return distance_dense(vec1, vec2, metric, data_type, quantize_type)


def calculate_rrf_score(rank, k=60):
    return 1.0 / (k + rank + 1)


def calculate_multi_vector_rrf_scores(query_results: Dict[str, Doc], k=60):
    rrf_scores = {}

    for vector_name, docs in query_results.items():
        for rank, doc in enumerate(docs):
            doc_id = doc.id
            rrf_score = calculate_rrf_score(rank, k)
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += rrf_score
            else:
                rrf_scores[doc_id] = rrf_score

    return rrf_scores


def calculate_multi_vector_weighted_scores(
    query_results: Dict[str, Doc], weights: Dict[str, float], metric: MetricType
):
    def _normalize_score(score: float, metric: MetricType) -> float:
        if metric == MetricType.L2:
            return 1.0 - 2 * math.atan(score) / math.pi
        if metric == MetricType.IP:
            return 0.5 + math.atan(score) / math.pi
        if metric == MetricType.COSINE:
            return 1.0 - score / 2.0
        raise ValueError("Unsupported metric type")

    weighted_scores = {}

    for vector_name, docs in query_results.items():
        weight = weights.get(vector_name, 1.0)

        for doc in docs:
            doc_id = doc.id
            weighted_score = (_normalize_score(doc.score, metric)) * weight
            if doc_id in weighted_scores:
                weighted_scores[doc_id] += weighted_score
            else:
                weighted_scores[doc_id] = weighted_score

    return weighted_scores


def is_field_equal(field1, field2, schema: FieldSchema) -> bool:
    if field1 is None and field2 is None:
        return True
    if field1 is None or field2 is None:
        return False

    if schema.data_type == DataType.ARRAY_FLOAT:
        return is_float_array_equal(field1, field2)
    elif schema.data_type == DataType.ARRAY_DOUBLE:
        return is_double_array_equal(field1, field2)
    elif schema.data_type in [
        DataType.ARRAY_INT32,
        DataType.ARRAY_INT64,
        DataType.ARRAY_BOOL,
        DataType.ARRAY_STRING,
        DataType.ARRAY_UINT32,
        DataType.ARRAY_UINT64,
        DataType.ARRAY_INT64,
    ]:
        return is_int_array_equal(field1, field2)
    elif schema.data_type in [DataType.FLOAT, DataType.DOUBLE]:
        return is_float_equal(field1, field2)

    return field1 == field2


def is_vector_equal(vec1, vec2, schema: VectorSchema) -> bool:
    if (
        schema.data_type == DataType.SPARSE_VECTOR_FP16
        or schema.data_type == DataType.VECTOR_FP16
    ):
        # skip fp16 vector equal
        return True

    is_sparse = (
        schema.data_type == DataType.SPARSE_VECTOR_FP32
        or schema.data_type == DataType.SPARSE_VECTOR_FP16
    )

    if is_sparse:
        return is_sparse_vector_equal(vec1, vec2)
    else:
        return is_dense_vector_equal(vec1, vec2)


def is_doc_equal(
    doc1: Doc,
    doc2: Doc,
    schema: CollectionSchema,
    except_score: bool = True,
    include_vector: bool = True,
):
    if doc1.id != doc2.id:
        logging.error("doc ids are not equal")
        return False

    reduce_field_names = set(doc1.field_names() + doc2.field_names())
    reduce_vector_names = set(doc1.vector_names() + doc2.vector_names())

    is_doc1_fields_empty = doc1.fields is None or doc1.fields == {}
    is_doc2_fields_empty = doc2.fields is None or doc2.fields == {}

    if is_doc1_fields_empty or is_doc2_fields_empty:
        if is_doc1_fields_empty != is_doc2_fields_empty:
            return False
    else:
        for field_name in reduce_field_names:
            field_schema = schema.field(field_name)
            if field_schema is None:
                return False
            if is_field_equal(
                doc1.field(field_name), doc2.field(field_name), field_schema
            ):
                continue
            else:
                logging.error(f"{field_name} are not equal")
                return False

    if include_vector:
        is_doc1_vectors_empty = doc1.vectors is None or doc1.vectors == {}
        is_doc2_vectors_empty = doc2.vectors is None or doc2.vectors == {}

        if is_doc1_vectors_empty or is_doc2_vectors_empty:
            if is_doc1_fields_empty != is_doc2_vectors_empty:
                return False
        else:
            for vector_name in reduce_vector_names:
                vector_schema = schema.vector(vector_name)
                if vector_schema is None:
                    return False
                if is_vector_equal(
                    doc1.vector(vector_name), doc2.vector(vector_name), vector_schema
                ):
                    continue
                else:
                    return False

    return True
