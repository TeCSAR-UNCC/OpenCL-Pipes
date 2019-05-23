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



FPGA_SRC = bfs_fpga.cpp clutils_fpga.cpp utils.cpp timer.cc ./util/opencl/opencl.o



all: bfs_fpga.exe


bfs_fpga.exe: $(FPGA_SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(FPGA_SRC) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

opencl.o: ./util/opencl/opencl.c ./util/opencl/opencl.h
	gcc $< -o ./util/opencl/opencl.o -O3

clean: $(SRC)
	rm -f *.exe result*
