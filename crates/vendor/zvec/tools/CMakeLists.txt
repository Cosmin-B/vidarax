include(${CMAKE_SOURCE_DIR}/cmake/bazel.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/option.cmake)

# Retrieve version from git repository
git_version(ZVEC_VERSION ${CMAKE_CURRENT_SOURCE_DIR})

# Add repository
cc_directory(core)