project(mnist LANGUAGES CXX CUDA)

file(GLOB_RECURSE PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)

add_executable(${PROJECT_NAME} ${PROJECT_SRCS})

target_link_libraries(${PROJECT_NAME} nn)
