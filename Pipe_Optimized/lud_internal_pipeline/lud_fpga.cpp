#include "lud.h"
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include <sys/time.h>
#include <CL/cl.h>
#include <string.h>
#include <string>

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
	
#endif


cl_context context=NULL;


static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};





int main(int argc, char *argv[]) {
	

   printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m;
	float *recordDistances;
	
	while ((opt = getopt_long(argc, argv, "::vs:i:", 
                            long_options, &option_index)) != -1 ) {
		switch(opt){
			case 'i':
			input_file = optarg;
			break;

        case 's':
			matrix_dim = atoi(optarg);
			printf("Generate input matrix internally, size =%d\n", matrix_dim);

			break;
        case '?':
			fprintf(stderr, "invalid option\n");
			break;
        case ':':
			fprintf(stderr, "missing argument\n");
			break;
        default:
			fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                  argv[0]);
			exit(EXIT_FAILURE);
		}
	}
  
	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}	

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	} 
	
	else if (matrix_dim) {
	  printf("Creating matrix internally size=%d\n", matrix_dim);
	  ret = create_matrix(&m, matrix_dim);
	  if (ret != RET_SUCCESS) {
	    m = NULL;
	    fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
	    exit(EXIT_FAILURE);
	  }
	}

	else {
	  printf("No input file specified!\n");
	  exit(EXIT_FAILURE);
	}


	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }
	
    int quiet=0,timing=0,platform=-1,device=-1;  
 


  if (!quiet) {
    printf("Number of records: %d\n",sourcesize);

  context = cl_init_context(platform,device,quiet);
  
  recordDistances = OpenCl_LUD(context,sourcesize, matrix_dim, m, timing); 

  free(recordDistances);
  return 0;

}



float *OpenCl_LUD(cl_context context,int sourcesize,int matrix_dim,	float *m, int timing) {




		cl_kernel lud_internal_read,lud_internal_compute, lud_internal_wb;
        cl_int status;
        float writeTime=0, kernel_1=0, readTime=0, kernel_2=0;
        cl_program cl_NN_program;
        cl_NN_program = cl_compileProgram((char *)"./xclbin/lud_diagonal.hw.xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xclbin",NULL);
	   

	   
        lud_internal_read = clCreateKernel(cl_NN_program, "lud_internal_read", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_read",true);
        if(status)exit(1);

		lud_internal_compute = clCreateKernel(cl_NN_program, "lud_internal_compute", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_compute",true);
        if(status)exit(1);    
                     
		lud_internal_wb = clCreateKernel(cl_NN_program, "lud_internal_wb", &status);
        status = cl_errChk(status, (char *)"Error Creating lud kernel_wb",true);
        if(status)exit(1);
    
    cl_mem d_m; //define var
    
    cl_int error=0;
	
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &error); // allocate mem

    cl_command_queue command_queue =  cl_getCommandQueue(); //command queues for three parallel kernels (read, comp, write)
	cl_command_queue command_queue1 = cl_getCommandQueue1(); 
	cl_command_queue command_queue2 = cl_getCommandQueue2();
	
	printf("created command Q \n");
		   
	cl_event writeEvent,kernelEvent,kernelEvent1,kernelEvent2,readEvent;
	
	error = clEnqueueWriteBuffer(command_queue,d_m,1,0,matrix_dim*matrix_dim*sizeof(float), m,0,0,&writeEvent); //Copy the var from host to device
	
	cl_errChk(error,"ERROR in clEnqueueWriteBuffer",true);							
	
	
	
	error=clFinish(command_queue);
	cl_errChk(error,"ERROR in command_queue",true);								  

printf("Host to device copy over\n");

    int i=0;
    printf("Matrix_dim=%d\n",matrix_dim);
    printf("Block dim=%d\n",BLOCK_SIZE);
         
    cl_int argchk;
	
    argchk  = clSetKernelArg(lud_internal_read, 0, sizeof(void *), (void*) &d_m);
    argchk |= clSetKernelArg(lud_internal_read, 1, sizeof(cl_int), (void*) &matrix_dim);
	argchk |= clSetKernelArg(lud_internal_read, 2, sizeof(cl_int), (void*) &i);
	
	 size_t global_work1[3]  = {BLOCK_SIZE, 1, 1};
	 size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};
	cl_errChk(argchk,"ERROR in Setting lud_read kernel args",true);	     
 

      argchk = clSetKernelArg(lud_internal_compute, 0, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL);
	  argchk |= clSetKernelArg(lud_internal_compute, 1, sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, (void*)NULL);
	  argchk |=clSetKernelArg(lud_internal_compute, 2, sizeof(cl_int), (void*) &matrix_dim );
	  argchk |=clSetKernelArg(lud_internal_compute, 3, sizeof(cl_int), (void*) &i );
	 cl_errChk(argchk,"ERROR in Setting lud_compute kernel args",true);   	 

	
	
      argchk = clSetKernelArg(lud_internal_wb, 0, sizeof(void *), (void*) &d_m);
	  argchk |= clSetKernelArg(lud_internal_wb, 1, sizeof(cl_int), (void*) &matrix_dim);
	  argchk |= clSetKernelArg(lud_internal_wb, 2, sizeof(cl_int), (void*) &i);
	
	cl_errChk(argchk,"ERROR in Setting lud_WB kernel args",true);
    
	error = clEnqueueNDRangeKernel(command_queue,lud_internal_read,2, NULL, global_work1, local_work1, 0, 0, &kernelEvent);

	cl_errChk(error,"ERROR in Executing Kernel lud_read kernel",true);


    error = clEnqueueNDRangeKernel(command_queue1,lud_internal_compute, 2, NULL, global_work1, local_work1, 0, 0, &kernelEvent1);
	cl_errChk(error,"ERROR in Executing Kernel lud_compute kernel",true);

    error = clEnqueueNDRangeKernel(command_queue2,lud_internal_wb, 2, NULL, global_work1, local_work1, 0, 0, &kernelEvent2);
    cl_errChk(error,"ERROR in Executing Kernel lud_wb kernel",true);

    
	status = clFinish(command_queue);
	status |= clFinish(command_queue1);
	status |= clFinish(command_queue2);
	cl_errChk(status,"ERROR with ndrange finishing Q",true);

    
    error = clEnqueueReadBuffer( command_queue2,d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0,0,&readEvent);
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
	
	status = clFinish(command_queue2);
	cl_errChk(status,"ERROR with Read buffer finishing Q",true);
	printf("Read final results from Device\n");
    
    
     if (1) {
        clFinish( command_queue2);
        cl_ulong eventStart,eventEnd,totalTime=0;
        printf("# Records\tWrite(s) [size]\t\tKernel_Read(s)\tKernel_Compute(s)\tKernel_Wb(s)\tRead(s)  [size]\t\tTotal(s)\n");
        printf("%d        \t",sourcesize);

        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write Start)",true); 
        error = clGetEventProfilingInfo(writeEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Write End)",true);


        totalTime += eventEnd-eventStart;

        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;

        error = clGetEventProfilingInfo(kernelEvent1,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent1,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;

        error = clGetEventProfilingInfo(kernelEvent2,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel Start)",true); 
        error = clGetEventProfilingInfo(kernelEvent2,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Kernel End)",true);

        printf("%f\t",(float)((eventEnd-eventStart)/1e9));
        totalTime += eventEnd-eventStart;

        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_START,
                                        sizeof(cl_ulong),&eventStart,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read Start)",true); 
        error = clGetEventProfilingInfo(readEvent,CL_PROFILING_COMMAND_END,
                                        sizeof(cl_ulong),&eventEnd,NULL);
        cl_errChk(error,"ERROR in Event Profiling (Read End)",true);


        totalTime += eventEnd-eventStart;
        
        printf("%f\n\n",(float)(totalTime/1e9));
        
    }
    

    clReleaseMemObject(d_m);
    
   	free(m);
	
	

	return 0;	 
}


