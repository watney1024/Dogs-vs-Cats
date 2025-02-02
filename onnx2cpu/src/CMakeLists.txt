
message(">>>>> ${PROJECT_SOURCE_DIR}")


include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(demo main.cpp func.cpp)

target_link_libraries(demo )