include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)
#message(STATUS "PROJECT_ROOT_DIR = ${PROJECT_ROOT_DIR}")
cc_library(
    NAME core_knn_flat 
    STATIC SHARED STRICT ALWAYS_LINK
    SRCS *.cc
    LIBS core_framework 
    INCS . ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm ${PROJECT_ROOT_DIR}/src/core/framework
    VERSION "${PROXIMA_ZVEC_VERSION}"
  )
