COMMON_REPO =../../../

CC = g++
CC_FLAGS = -g -O3 -Wall

# XILINX_OPENCL
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/bitmap/bitmap.mk

XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64

FPGA_SRC = hotspot_fpga.cpp hotspot_fpga.h timer.cc clutils_fpga.cpp utils.cpp ./util/opencl/opencl.o

all:hotspot_fpga.exe

hotspot_fpga.exe: $(FPGA_SRC) 
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

clean:
	rm -f *.exe 
