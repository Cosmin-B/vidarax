import pytest
import logging

from typing import Any, Generator

import zvec
from zvec import (
    CollectionOption,
    InvertIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    FieldSchema,
    VectorSchema,
    CollectionSchema,
    Collection,
    Doc,
    VectorQuery,
)

from support_helper import *


@pytest.fixture(scope="session")
def basic_schema(collection_name="test_collection") -> CollectionSchema:
    return CollectionSchema(
        name=collection_name if len(collection_name) > 0 else "test_collection",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
            FieldSchema(
                "name", DataType.STRING, nullable=False, index_param=InvertIndexParam()
            ),
            FieldSchema("weight", DataType.FLOAT, nullable=True),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=HnswIndexParam(),
            ),
            VectorSchema(
                "sparse", DataType.SPARSE_VECTOR_FP32, index_param=HnswIndexParam()
            ),
        ],
    )


@pytest.fixture(scope="session")
def full_schema(
    nullable: bool = False,
    has_index: bool = False,
) -> CollectionSchema:
    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = HnswIndexParam()

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vetors = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        vetors.append(
            VectorSchema(
                v,
                k,
                dimension=DEFAULT_VECTOR_DIMENSION,
                index_param=vector_index_param,
            )
        )

    return CollectionSchema(
        name="full_collection",
        fields=fields,
        vectors=vetors,
    )


@pytest.fixture(scope="function")
def full_schema_new(request) -> CollectionSchema:
    if hasattr(request, "param"):
        nullable, has_index, vector_index = request.param
    else:
        nullable, has_index, vector_index = True, False, HnswIndexParam()

    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = vector_index

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vectors = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        vectors.append(
            VectorSchema(
                v,
                k,
                dimension=DEFAULT_VECTOR_DIMENSION,
                index_param=vector_index_param,
            )
        )

    return CollectionSchema(
        name="full_collection_new",
        fields=fields,
        vectors=vectors,
    )


@pytest.fixture(scope="function")
def full_schema_ivf(request) -> CollectionSchema:
    if hasattr(request, "param"):
        nullable, has_index, vector_index = request.param
    else:
        nullable, has_index, vector_index = True, False, IVFIndexParam()

    scalar_index_param = None
    vector_index_param = None
    if has_index:
        scalar_index_param = InvertIndexParam(enable_range_optimization=True)
        vector_index_param = vector_index

    fields = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        fields.append(
            FieldSchema(
                v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    vectors = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        if v in ["vector_fp16_field", "vector_fp32_field"]:
            vectors.append(
                VectorSchema(
                    v,
                    k,
                    dimension=DEFAULT_VECTOR_DIMENSION,
                    index_param=vector_index_param,
                )
            )

    return CollectionSchema(
        name="full_collection_ivf",
        fields=fields,
        vectors=vectors,
    )


@pytest.fixture(scope="function")
def single_vector_schema(
    data_type: DataType,
) -> CollectionSchema:
    vector_schema = [
        VectorSchema(
            DEFAULT_VECTOR_FIELD_NAME[data_type],
            data_type,
            DEFAULT_VECTOR_DIMENSION,
        )
    ]

    return CollectionSchema(
        name="full_collection",
        vectors=vector_schema,
    )


@pytest.fixture(scope="function")
def single_vector_schema_with_index_param(
    data_type: DataType, index_param
) -> CollectionSchema:
    vector_schema = [
        VectorSchema(
            DEFAULT_VECTOR_FIELD_NAME[data_type],
            data_type,
            DEFAULT_VECTOR_DIMENSION,
            index_param,
        )
    ]

    return CollectionSchema(
        name="full_collection",
        vectors=vector_schema,
    )


def create_collection_fixture(
    collection_temp_dir, schema: CollectionSchema, collection_option: CollectionOption
) -> Generator[Any, Any, Collection]:
    """Common helper function to create and manage collection fixtures."""
    coll = zvec.create_and_open(
        path=str(collection_temp_dir),
        schema=schema,
        option=collection_option,
    )

    assert coll is not None, "Failed to create and open collection"
    assert coll.path == str(collection_temp_dir)
    assert coll.schema.name == schema.name
    assert list(coll.schema.fields) == list(schema.fields)
    assert list(coll.schema.vectors) == list(schema.vectors)
    assert coll.option.read_only == collection_option.read_only
    assert coll.option.enable_mmap == collection_option.enable_mmap

    try:
        yield coll
    finally:
        if hasattr(coll, "destroy") and coll is not None:
            try:
                coll.destroy()
            except Exception as e:
                logging.warning(f"Warning: failed to destroy collection: {e}")


@pytest.fixture(scope="function")
def basic_collection(
    collection_temp_dir, basic_schema, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, basic_schema, collection_option
    )


@pytest.fixture(scope="function")
def collection_option():
    return CollectionOption(read_only=False, enable_mmap=True)


@pytest.fixture(scope="function")
def collection_temp_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("zvec")
    collection_path = temp_dir / "test_collection_path"
    return str(collection_path)


@pytest.fixture(scope="function")
def full_collection(
    collection_temp_dir,
    full_schema,
    collection_option,
    nullable: bool = True,
    has_index: bool = False,
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema, collection_option
    )


@pytest.fixture(scope="function")
def full_collection_new(
    collection_temp_dir, full_schema_new, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema_new, collection_option
    )


@pytest.fixture(scope="function")
def full_collection_ivf(
    collection_temp_dir, full_schema_ivf, collection_option
) -> Generator[Any, Any, Collection]:
    yield from create_collection_fixture(
        collection_temp_dir, full_schema_ivf, collection_option
    )


@pytest.fixture
def sample_field_list(nullable: bool = True, scalar_index_param=None, name_prefix=""):
    field_list = []
    for k, v in DEFAULT_SCALAR_FIELD_NAME.items():
        field_list.append(
            FieldSchema(
                f"{name_prefix}_{v}" if len(name_prefix) > 0 else v,
                k,
                nullable=nullable,
                index_param=scalar_index_param,
            )
        )
    return field_list


@pytest.fixture
def sample_vector_list(vector_index_param=None, name_prefix=""):
    vector_list = []
    for k, v in DEFAULT_VECTOR_FIELD_NAME.items():
        vector_list.append(
            VectorSchema(
                f"{name_prefix}_{v}" if len(name_prefix) > 0 else v,
                k,
                dimension=DEFAULT_VECTOR_DIMENSION,
                index_param=vector_index_param,
            )
        )
    return vector_list
