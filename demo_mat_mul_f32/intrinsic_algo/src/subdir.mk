################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/demo_mat_mul_f32.c \
../src/mat_mul_f32.c \
../src/nds_version.c 

OBJS += \
./src/demo_mat_mul_f32.o \
./src/mat_mul_f32.o \
./src/nds_version.o 

C_DEPS += \
./src/demo_mat_mul_f32.d \
./src/mat_mul_f32.d \
./src/nds_version.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Andes C Compiler'
	$(CROSS_COMPILE)clang -mext-vector -DENA_VEC_ISA -DENA_VEC_INTRINSIC -O3 -mcmodel=medium -fno-tree-ter -g3 -Wall -mcpu=nx27v -ffunction-sections -fdata-sections -c -mext-vector -fno-unroll-loops -fno-tree-slp-vectorize -fno-tree-vectorize -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d) $(@:%.o=%.o)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


