COMMON_REPO =../../../

include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/bitmap/bitmap.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

#C compiler
CC = g++

# Xilinx OpenCL
#XILINX_OPENCL := /opt/Xilinx/SDAccel/2015.4
XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

#SRC_GPU = nearestNeighbor_gpu.cpp clutils_gpu.cpp utils.cpp
SRC_FPGA = nearestNeighbor_fpga.cpp clutils_fpga.cpp utils.cpp

CC_FLAGS = -g -O3

all: nn_fpga.exe \

#nn_gpu.exe:
#	$(CC) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC_GPU) -o $@ -I
#$(OPENCL_INC) -L$(OPENCL_LIB)
	
nn_fpga.exe:
	$(CC) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC_FPGA) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB)

clean: 
	rm -f *.exe 