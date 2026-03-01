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
#pragma once

#include <string>

namespace zvec {
namespace core {

//! General
static const std::string GENERAL_CLUSTER_COUNT =
    "proxima.general.cluster.count";
static const std::string GENERAL_THREAD_COUNT =
    "proxima.general.cluster.thread_count";

//! Optimize K-means
static const std::string OPTKMEANS_CLUSTER_COUNT =
    "proxima.optkmeans.cluster.count";
static const std::string OPTKMEANS_CLUSTER_MAX_ITERATIONS =
    "proxima.optkmeans.cluster.max_iterations";
static const std::string OPTKMEANS_CLUSTER_EPSILON =
    "proxima.optkmeans.cluster.epsilon";
static const std::string OPTKMEANS_CLUSTER_SHARD_FACTOR =
    "proxima.optkmeans.cluster.shard_factor";
static const std::string OPTKMEANS_CLUSTER_PURGE_EMPTY =
    "proxima.optkmeans.cluster.purge_empty";
static const std::string OPTKMEANS_CLUSTER_MARKOV_CHAIN_LENGTH =
    "proxima.optkmeans.cluster.markov_chain_length";
static const std::string OPTKMEANS_CLUSTER_ASSUMPTION_FREE =
    "proxima.optkmeans.cluster.assumption_free";

//! K-means
static const std::string KMEANS_CLUSTER_COUNT = "proxima.kmeans.cluster.count";
static const std::string KMEANS_CLUSTER_SHARD_FACTOR =
    "proxima.kmeans.cluster.shard_factor";
static const std::string KMEANS_CLUSTER_EPSILON =
    "proxima.kmeans.cluster.epsilon";
static const std::string KMEANS_CLUSTER_MAX_ITERATIONS =
    "proxima.kmeans.cluster.max_iterations";
static const std::string KMEANS_CLUSTER_PURGE_EMPTY =
    "proxima.kmeans.cluster.purge_empty";
static const std::string KMEANS_CLUSTER_BATCH = "proxima.kmeans.cluster.batch";
static const std::string KMEANS_CLUSTER_SEEKER_CLASS =
    "proxima.kmeans.cluster.seeker_class";
static const std::string KMEANS_CLUSTER_SEEKER_PARAMS =
    "proxima.kmeans.cluster.seeker_params";

//! Mini Batch K-means
static const std::string MINIBATCHKMEANS_CLUSTER_COUNT =
    "proxima.minibatchkmeans.cluster.count";
static const std::string MINIBATCHKMEANS_CLUSTER_SHARD_FACTOR =
    "proxima.minibatchkmeans.cluster.shard_factor";
static const std::string MINIBATCHKMEANS_CLUSTER_EPSILON =
    "proxima.minibatchkmeans.cluster.epsilon";
static const std::string MINIBATCHKMEANS_CLUSTER_MAX_ITERATIONS =
    "proxima.minibatchkmeans.cluster.max_iterations";
static const std::string MINIBATCHKMEANS_CLUSTER_PURGE_EMPTY =
    "proxima.minibatchkmeans.cluster.purge_empty";
static const std::string MINIBATCHKMEANS_CLUSTER_TRY_COUNT =
    "proxima.minibatchkmeans.cluster.try_count";
static const std::string MINIBATCHKMEANS_CLUSTER_BATCH_COUNT =
    "proxima.minibatchkmeans.cluster.batch_count";
static const std::string MINIBATCHKMEANS_CLUSTER_SEEKER_CLASS =
    "proxima.minibatchkmeans.cluster.seeker_class";
static const std::string MINIBATCHKMEANS_CLUSTER_SEEKER_PARAMS =
    "proxima.minibatchkmeans.cluster.seeker_params";

//! K-means++
static const std::string KMEANSPP_CLUSTER_COUNT =
    "proxima.kmeanspp.cluster.count";
static const std::string KMEANSPP_CLUSTER_SHARD_FACTOR =
    "proxima.kmeanspp.cluster.shard_factor";
static const std::string KMEANSPP_CLUSTER_CLASS =
    "proxima.kmeanspp.cluster.class";
static const std::string KMEANSPP_CLUSTER_PARAMS =
    "proxima.kmeanspp.cluster.params";

//! K-MC2
static const std::string KMC2_CLUSTER_COUNT = "proxima.kmc2.cluster.count";
static const std::string KMC2_CLUSTER_SHARD_FACTOR =
    "proxima.kmc2.cluster.shard_factor";
static const std::string KMC2_CLUSTER_MARKOV_CHAIN_LENGTH =
    "proxima.kmc2.cluster.markov_chain_length";
static const std::string KMC2_CLUSTER_ASSUMPTION_FREE =
    "proxima.kmc2.cluster.assumption_free";
static const std::string KMC2_CLUSTER_CLASS = "proxima.kmc2.cluster.class";
static const std::string KMC2_CLUSTER_PARAMS = "proxima.kmc2.cluster.params";

//! Bisecting K-means
static const std::string BIKMEANS_CLUSTER_COUNT =
    "proxima.bikmeans.cluster.count";
static const std::string BIKMEANS_CLUSTER_INIT_COUNT =
    "proxima.bikmeans.cluster.init_count";
static const std::string BIKMEANS_CLUSTER_PURGE_EMPTY =
    "proxima.bikmeans.cluster.purge_empty";
static const std::string BIKMEANS_CLUSTER_FIRST_CLASS =
    "proxima.bikmeans.cluster.first_class";
static const std::string BIKMEANS_CLUSTER_SECOND_CLASS =
    "proxima.bikmeans.cluster.second_class";
static const std::string BIKMEANS_CLUSTER_FIRST_PARAMS =
    "proxima.bikmeans.cluster.first_params";
static const std::string BIKMEANS_CLUSTER_SECOND_PARAMS =
    "proxima.bikmeans.cluster.second_params";

//! K-medoids
static const std::string KMEDOIDS_CLUSTER_COUNT =
    "proxima.kmedoids.cluster.count";
static const std::string KMEDOIDS_CLUSTER_SHARD_FACTOR =
    "proxima.kmedoids.cluster.shard_factor";
static const std::string KMEDOIDS_CLUSTER_EPSILON =
    "proxima.kmedoids.cluster.epsilon";
static const std::string KMEDOIDS_CLUSTER_MAX_ITERATIONS =
    "proxima.kmedoids.cluster.max_iterations";
static const std::string KMEDOIDS_CLUSTER_PURGE_EMPTY =
    "proxima.kmedoids.cluster.purge_empty";
static const std::string KMEDOIDS_CLUSTER_BENCH_RATIO =
    "proxima.kmedoids.cluster.bench_ratio";
static const std::string KMEDOIDS_CLUSTER_ONLY_MEANS =
    "proxima.kmedoids.cluster.only_means";
static const std::string KMEDOIDS_CLUSTER_WITHOUT_MEANS =
    "proxima.kmedoids.cluster.without_means";
static const std::string KMEDOIDS_CLUSTER_SEEKER_CLASS =
    "proxima.kmedoids.cluster.seeker_class";
static const std::string KMEDOIDS_CLUSTER_SEEKER_PARAMS =
    "proxima.kmedoids.cluster.seeker_params";

//! Stratified
static const std::string STRATIFIED_CLUSTER_COUNT =
    "proxima.stratified.cluster.count";
static const std::string STRATIFIED_CLUSTER_FIRST_CLASS =
    "proxima.stratified.cluster.first_class";
static const std::string STRATIFIED_CLUSTER_SECOND_CLASS =
    "proxima.stratified.cluster.second_class";
static const std::string STRATIFIED_CLUSTER_FIRST_COUNT =
    "proxima.stratified.cluster.first_count";
static const std::string STRATIFIED_CLUSTER_SECOND_COUNT =
    "proxima.stratified.cluster.second_count";
static const std::string STRATIFIED_CLUSTER_FIRST_PARAMS =
    "proxima.stratified.cluster.first_params";
static const std::string STRATIFIED_CLUSTER_SECOND_PARAMS =
    "proxima.stratified.cluster.second_params";
static const std::string STRATIFIED_CLUSTER_AUTO_TUNING =
    "proxima.stratified.cluster.auto_tuning";
static const std::string STRATIFIED_CLUSTER_SECOND_POOL_COUNT =
    "proxima.stratified.cluster.second_pool_count";

//! Gap Statistics
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MIN =
    "proxima.gapstats.cluster_estimater.k_min";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MAX =
    "proxima.gapstats.cluster_estimater.k_max";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MIN_STEP =
    "proxima.gapstats.cluster_estimater.k_min_step";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_K_MAX_STEP =
    "proxima.gapstats.cluster_estimater.k_max_step";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_TRY_COUNT =
    "proxima.gapstats.cluster_estimater.try_count";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_SHARD_FACTOR =
    "proxima.gapstats.cluster_estimater.shard_factor";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_ENABLE_MC2 =
    "proxima.gapstats.cluster_estimater.enable_mc2";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_MARKOV_CHAIN_LENGTH =
    "proxima.gapstats.cluster_estimater.markov_chain_length";
static const std::string GAPSTATS_CLUSTER_ESTIMATER_CLUSTER_CLASS =
    "proxima.gapstats.cluster_estimater.cluster_class";

static const std::string CLUSTER_TRAINER_SAMPLE_COUNT =
    "proxima.cluster.trainer.sample_count";
static const std::string CLUSTER_TRAINER_SAMPLE_RATIO =
    "proxima.cluster.trainer.sample_ratio";
static const std::string CLUSTER_TRAINER_THREAD_COUNT =
    "proxima.cluster.trainer.thread_count";
static const std::string CLUSTER_TRAINER_FILE_NAME =
    "proxima.cluster.trainer.file_name";
static const std::string CLUSTER_TRAINER_CLASS_NAME =
    "proxima.cluster.trainer.class_name";

static const std::string STRATIFIED_TRAINER_SAMPLE_COUNT =
    "proxima.stratified.trainer.sample_count";
static const std::string STRATIFIED_TRAINER_SAMPLE_RATIO =
    "proxima.stratified.trainer.sample_ratio";
static const std::string STRATIFIED_TRAINER_THREAD_COUNT =
    "proxima.stratified.trainer.thread_count";
static const std::string STRATIFIED_TRAINER_FILE_NAME =
    "proxima.stratified.trainer.file_name";
static const std::string STRATIFIED_TRAINER_CLASS_NAME =
    "proxima.stratified.trainer.class_name";
static const std::string STRATIFIED_TRAINER_CLUSTER_COUNT =
    "proxima.stratified.trainer.cluster_count";
static const std::string STRATIFIED_TRAINER_AUTOAUNE =
    "proxima.stratified.trainer.autotune";
static const std::string STRATIFIED_TRAINER_PARAMS_IN_LEVEL_PREFIX =
    "proxima.stratified.trainer.cluster_params_in_level_";

}  // namespace core
}  // namespace zvec
