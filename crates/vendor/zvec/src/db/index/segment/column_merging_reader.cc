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

#include "column_merging_reader.h"
#include <iostream>
#include <arrow/array.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include "db/index/storage/store_helper.h"

namespace zvec {

std::shared_ptr<ColumnMergingReader> ColumnMergingReader::Make(
    const std::shared_ptr<arrow::Schema> &target_schema,
    std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>>
        &&input_readers) {
  return std::make_shared<ColumnMergingReader>(target_schema,
                                               std::move(input_readers));
}

ColumnMergingReader::ColumnMergingReader(
    const std::shared_ptr<arrow::Schema> &target_schema,
    std::vector<std::shared_ptr<arrow::ipc::RecordBatchReader>> &&input_readers)
    : target_schema_(target_schema), input_readers_(std::move(input_readers)) {
  current_batches_.resize(input_readers_.size());
  std::fill(current_batches_.begin(), current_batches_.end(), nullptr);
}

std::shared_ptr<arrow::Schema> ColumnMergingReader::schema() const {
  return target_schema_;
}

arrow::Status ColumnMergingReader::ReadNext(
    std::shared_ptr<arrow::RecordBatch> *out) {
  *out = nullptr;

  if (!has_more_) {
    return arrow::Status::OK();
  }

  // Read next batch from each input reader
  for (size_t i = 0; i < input_readers_.size(); ++i) {
    arrow::Status status = input_readers_[i]->ReadNext(&current_batches_[i]);
    if (!status.ok()) {
      return status;
    }
  }

  // Check if all readers have reached EOF
  bool all_null = true;
  for (const auto &batch : current_batches_) {
    if (batch != nullptr) {
      all_null = false;
      break;
    }
  }

  // All readers reached EOF
  if (all_null) {
    has_more_ = false;
    return arrow::Status::OK();
  }

  // Verify that all non-null batches have consistent row counts
  int64_t expected_rows = -1;
  for (const auto &batch : current_batches_) {
    if (batch) {
      if (expected_rows == -1) {
        expected_rows = batch->num_rows();
      } else if (expected_rows != batch->num_rows()) {
        return arrow::Status::Invalid(
            "Input readers have inconsistent row counts");
      }
    }
  }

  if (expected_rows <= 0) {
    has_more_ = false;
    return arrow::Status::OK();
  }

  // Build each column
  std::vector<std::shared_ptr<arrow::Array>> columns;
  columns.reserve(target_schema_->num_fields());

  for (int i = 0; i < target_schema_->num_fields(); ++i) {
    auto field = target_schema_->field(i);
    std::shared_ptr<arrow::Array> col_array = nullptr;

    // Try to find this column from any batch
    for (const auto &batch : current_batches_) {
      if (!batch) continue;
      int col_idx = batch->schema()->GetFieldIndex(field->name());
      if (col_idx != -1) {
        col_array = batch->column(col_idx);
        break;
      }
    }

    if (!col_array) {
      return arrow::Status::Invalid(
          "Failed to find column in any input reader: ", field->name());
    }

    columns.push_back(std::move(col_array));
  }

  // Construct final batch
  *out = arrow::RecordBatch::Make(target_schema_, expected_rows,
                                  std::move(columns));
  if (!*out) {
    return arrow::Status::Invalid("Failed to create merged record batch");
  }

  // Clear current batches, prepare for next read
  std::fill(current_batches_.begin(), current_batches_.end(), nullptr);

  return arrow::Status::OK();
}

}  // namespace zvec