project(nn-test LANGUAGES CXX CUDA)

file(GLOB_RECURSE PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.cu)
#file(GLOB_RECURSE PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)

add_executable(${PROJECT_NAME} ${PROJECT_SRCS})

#target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(GTest REQUIRED)
target_link_libraries(${PROJECT_NAME} ${GTEST_LIBRARIES})

target_link_libraries(${PROJECT_NAME} nn)
