//========================================================================================================================================================================================================200
//	MAIN HEADER
//========================================================================================================================================================================================================200

//====================================================================================================100
//	DEFINE
//====================================================================================================100

#define fp float

#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 256
#endif


//====================================================================================================100
//	End
//====================================================================================================100

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200



pipe float p0 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p1 __attribute__((xcl_reqd_pipe_depth(512)));


__kernel void extract_kernel_read(__global fp* d_I){


	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;  // unique thread id, more threads than actual elements !!!
	
	write_pipe_block(p0, &d_I[ei]);
	
	
}

__kernel void extract_kernel_compute(long d_Ne){


	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;      // unique thread id, more threads than actual elements !!!
	
	float var_dI;
	read_pipe_block(p0, &var_dI);
	
	
	if(ei<d_Ne){			      // do only for the number of elements, omit extra threads

		var_dI = exp(var_dI/255);   // exponentiate input IMAGE and copy to output image

	}
	
	write_pipe_block(p1, &var_dI);
	}



__kernel void extract_kernel_wb(__global fp* d_I){	      // pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
	int bx = get_group_id(0);	      // get current horizontal block index
	int tx = get_local_id(0);	      // get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;      // unique thread id, more threads than actual elements !!!
	

	read_pipe_block(p1, &d_I[ei]); 
	
}
