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

#include "db/sqlengine/analyzer/query_info.h"

namespace zvec::sqlengine {

class SimpleRewriter {
 public:
  SimpleRewriter() = default;

  //! Rewrite query_info->search_cond and simplify tree
  void rewrite(QueryInfo *query_info);

 private:
  void simplify_tree(QueryNode::Ptr query_node, QueryInfo *query_info);
};

class RewriteRule {
 public:
  RewriteRule() = default;
  //! Rewrite filter, return whether successfully rewrited.
  virtual bool rewrite(QueryNode::Ptr query_node) = 0;

 protected:
  bool rewrited_{false};
};

class EqualOrRewriteRule : public RewriteRule {
 public:
  EqualOrRewriteRule() = default;

  bool rewrite(QueryNode::Ptr query_node) override;

 private:
  void rewrite_impl(bool is_or, QueryNode::Ptr query_node);

 private:
  QueryNode::Ptr cur_;
};

// ContainRewriteRule rewrites contain_all/any ()
class ContainRewriteRule : public RewriteRule {
 public:
  ContainRewriteRule() = default;

  bool rewrite(QueryNode::Ptr query_node) override;
};

}  // namespace zvec::sqlengine
