COMMON_REPO =../../../

include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/bitmap/bitmap.mk

XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

#C compiler
CC = g++
CC_FLAGS = -g2 -O3 -Wall



SRC_FPGA = lud_fpga.cpp clutils_fpga.cpp utils.cpp common.c

all: clean \
     lud_fpga.exe \
     

lud_fpga.exe:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -g -lOpenCL  -fopenmp $(SRC_FPGA) -o $@  -I$(OPENCL_INC) -L$(OPENCL_LIB) -I$(XILINX_OPENCL_INC) -L$(XILINX_OPENCL_LIB)
	

clean: 
	rm -f *.exe 
