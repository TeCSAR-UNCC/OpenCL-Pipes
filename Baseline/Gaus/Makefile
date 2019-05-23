COMMON_REPO =../../../

#C compiler
CC = g++
CC_FLAGS = -g -O3

# XILINX_OPENCL
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/bitmap/bitmap.mk

XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64


FPGA_SRC = clutils_fpga.cpp gaussianElim_fpga.cpp utils.cpp ./util/opencl/opencl.o



all: gaussian_fpga.exe


gaussian_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) 
opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean: 
	rm -f *.exe
