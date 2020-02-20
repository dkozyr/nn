project(nn LANGUAGES CXX CUDA)

file(GLOB_RECURSE PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.cu)

add_library(${PROJECT_NAME} STATIC ${PROJECT_SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)
enable_language(CUDA)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

#target_link_libraries(${PROJECT_NAME} boost)
