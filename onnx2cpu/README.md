## 
### 学习 CMake
#### CMake 自动设置变量
    CMake 提供了许多自动设置的变量
    【路径相关变量】
        CMAKE_SOURCE_DIR	指向 项目的根目录（即包含顶层 CMakeLists.txt 的目录）。
        CMAKE_BINARY_DIR	指向 构建目录（即 cmake .. 运行的目录）。
        PROJECT_SOURCE_DIR	指向 project() 命令所在的 CMakeLists.txt 文件目录，通常与 CMAKE_SOURCE_DIR 相同。
        PROJECT_BINARY_DIR	指向 project() 命令执行的二进制构建目录，通常与 CMAKE_BINARY_DIR 相同。
        CMAKE_CURRENT_SOURCE_DIR	指向当前 CMakeLists.txt 文件所在的目录（适用于子目录）。
        CMAKE_CURRENT_BINARY_DIR	指向当前处理的构建目录（适用于子目录）。
        CMAKE_MODULE_PATH	额外的 CMake 模块搜索路径。

    【目标文件、可执行文件相关】
        CMAKE_RUNTIME_OUTPUT_DIRECTORY	控制可执行文件（exe）的输出目录。
        CMAKE_LIBRARY_OUTPUT_DIRECTORY	控制动态库（*.so、*.dll）的输出目录。
        CMAKE_ARCHIVE_OUTPUT_DIRECTORY	控制静态库（*.a、*.lib）的输出目录。

    【系统与编译器相关】
        CMAKE_SYSTEM_NAME	目标系统的名称，如 Linux、Windows、Darwin（macOS）。
        CMAKE_SYSTEM_PROCESSOR	目标 CPU 架构，如 x86_64、ARM。
        CMAKE_CXX_COMPILER	C++ 编译器的路径。
        CMAKE_C_COMPILER	C 编译器的路径。
        CMAKE_CXX_FLAGS	C++ 编译器的额外参数。
        CMAKE_C_FLAGS	C 编译器的额外参数。

    【代码构建相关】
        CMAKE_BUILD_TYPE	构建类型，如 Debug、Release（只在单配置生成器中生效，如 Makefile）。
        CMAKE_VERBOSE_MAKEFILE	是否打印详细的编译过程（ON/OFF）。

    【项目 & 目标相关】
        PROJECT_NAME	project() 设定的项目名称。
        EXECUTABLE_OUTPUT_PATH	生成可执行文件的路径（已废弃，推荐 CMAKE_RUNTIME_OUTPUT_DIRECTORY）。
        LIBRARY_OUTPUT_PATH	生成库文件的路径（已废弃，推荐 CMAKE_LIBRARY_OUTPUT_DIRECTORY）。
        
    【其他常见变量】
        CMAKE_INSTALL_PREFIX	make install 目标的默认安装路径（通常是 /usr/local 或 C:/Program Files/）。
        CMAKE_PREFIX_PATH	额外的库搜索路径（常用于 find_package()）。
        CMAKE_MODULE_PATH	额外的 CMake 模块搜索路径。
#### 函数 和 功能
    CMake 基本函数
    set(): 设置变量的值。
        set(MY_VARIABLE "Some Value")
    project(): 定义项目的名称和使用的语言。
        project(MyProject VERSION 1.0 LANGUAGES CXX)
        VERSION：指定项目版本。
        LANGUAGES：指定支持的语言（如 CXX、C、CUDA 等）。
    add_executable(): 添加一个可执行文件目标。（之后的所有源文件只能有一个 main 函数）
        add_executable(MyApp src/main.cpp)
    add_library(): 添加一个库文件目标，可以是静态库或动态库。
        add_library(MyLibrary STATIC src/my_library.cpp)
        STATIC、SHARED、MODULE：指定库的类型。
    include_directories(): 指定一个或多个目录来包含头文件。（用来定义头文件位置）
        include_directories(${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(): 将目标与一个或多个库链接。
        target_link_libraries(MyApp MyLibrary)
    link_directories(): 指定一个或多个目录来搜索库文件。
        link_directories(${PROJECT_BINARY_DIR}/lib)



