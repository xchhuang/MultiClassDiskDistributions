cmake_minimum_required(VERSION 3.0)
project (DiskProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O3 -march=native -m64 -D_FORTIFY_SOURCE=2")


set(SOURCES
src/ASMCDD.cpp
src/utils.cpp
src/Category.cpp
src/computeFunctions.cpp
main.cpp
)

# 可执行程序中添加源文件
add_executable(DiskProject ${SOURCES})

target_include_directories(DiskProject PRIVATE  ${PROJECT_SOURCE_DIR}/include)


# Copy shaders to binary directory
file(GLOB shds RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/* )
foreach (s ${shds})
    configure_file(${s} ${s} COPYONLY)
endforeach ()

# Copy models to binary directory
file(GLOB meshes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} models/* )
foreach (s ${meshes})
    configure_file(${s} ${s} COPYONLY)
endforeach ()

# Copy examples to binary directory
file(GLOB expl RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} examples/* )
foreach (s ${expl})
    configure_file(${s} ${s} COPYONLY)
endforeach ()

# Copy configs to binary directory
file(GLOB cfgs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} configs/* )
foreach (s ${cfgs})
    configure_file(${s} ${s} COPYONLY)
endforeach ()
