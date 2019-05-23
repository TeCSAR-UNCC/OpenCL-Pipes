COMMON_REPO =../../../

#C compiler
CC = g++
CC_FLAGS = -g -O3
CC_FLAGS = -Wall
 
# XILINX_OPENCL
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/bitmap/bitmap.mk

XILINX_OPENCL_INC := $(XILINX_OPENCL)/runtime/include/1_2
XILINX_OPENCL_LIB := $(XILINX_OPENCL)/runtime/lib/x86_64


SRC_FPGA = srad_fpga.cpp clutils_fpga.cpp utils.cpp util/graphics/resize.c util/graphics/graphics.c util/opencl/opencl.c


CC_FLAGS = -g -O3
CC_FLAGS = -Wall

all: clean \
	 srad_fpga.exe \

     

srad_fpga.exe:
	$(CC) $(CC_FLAGS) -g -lOpenCL  -fopenmp $(SRC_FPGA) -o $@ -I$(OPENCL_INC) -L$(OPENCL_LIB)


clean: 
	rm -f *.exe 
