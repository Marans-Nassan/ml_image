# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(ml_image C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# Add TFLM library
add_subdirectory(pico-tflmicro)

# Add executable. Default name is the project name, version 0.1

add_executable(ml_image 
main.cc
lib/ssd1306.c )

pico_set_program_name(ml_image "ml_image")
pico_set_program_version(ml_image "0.1")

# Modify the below lines to enable/disable output over UART/USB
pico_enable_stdio_uart(ml_image 0)
pico_enable_stdio_usb(ml_image 1)

# Add the standard library to the build
target_link_libraries(ml_image
        pico_stdlib
        pico_multicore
        hardware_i2c
        pico-tflmicro)

# Add the standard include files to the build
target_include_directories(ml_image PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/lib
        ${CMAKE_CURRENT_LIST_DIR}/pico-tflmicro/src
)

# Add any user requested libraries
target_link_libraries(ml_image 
        
        )

pico_add_extra_outputs(ml_image)

