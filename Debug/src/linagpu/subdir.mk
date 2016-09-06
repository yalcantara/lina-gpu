################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/linagpu/CPUMatrix.cpp \
../src/linagpu/GPUMatrix.cpp \
../src/linagpu/Matrix.cpp \
../src/linagpu/utils.cpp 

OBJS += \
./src/linagpu/CPUMatrix.o \
./src/linagpu/GPUMatrix.o \
./src/linagpu/Matrix.o \
./src/linagpu/utils.o 

CPP_DEPS += \
./src/linagpu/CPUMatrix.d \
./src/linagpu/GPUMatrix.d \
./src/linagpu/Matrix.d \
./src/linagpu/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/linagpu/%.o: ../src/linagpu/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/linagpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


