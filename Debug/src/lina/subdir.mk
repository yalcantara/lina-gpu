################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/lina/CPUMatrix.cpp \
../src/lina/CPUVector.cpp \
../src/lina/Exception.cpp \
../src/lina/Func.cpp \
../src/lina/Grid.cpp \
../src/lina/GridColInfo.cpp \
../src/lina/GridInfo.cpp \
../src/lina/Layer.cpp \
../src/lina/Matrix.cpp \
../src/lina/Network.cpp \
../src/lina/Vector.cpp \
../src/lina/utils.cpp 

CU_SRCS += \
../src/lina/cudaUtils.cu 

CU_DEPS += \
./src/lina/cudaUtils.d 

OBJS += \
./src/lina/CPUMatrix.o \
./src/lina/CPUVector.o \
./src/lina/Exception.o \
./src/lina/Func.o \
./src/lina/Grid.o \
./src/lina/GridColInfo.o \
./src/lina/GridInfo.o \
./src/lina/Layer.o \
./src/lina/Matrix.o \
./src/lina/Network.o \
./src/lina/Vector.o \
./src/lina/cudaUtils.o \
./src/lina/utils.o 

CPP_DEPS += \
./src/lina/CPUMatrix.d \
./src/lina/CPUVector.d \
./src/lina/Exception.d \
./src/lina/Func.d \
./src/lina/Grid.d \
./src/lina/GridColInfo.d \
./src/lina/GridInfo.d \
./src/lina/Layer.d \
./src/lina/Matrix.d \
./src/lina/Network.d \
./src/lina/Vector.d \
./src/lina/utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/lina/%.o: ../src/lina/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/lina" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/lina/%.o: ../src/lina/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/lina" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


