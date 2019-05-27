#define BLOCK_SIZE 16


pipe float p0 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p1 __attribute__((xcl_reqd_pipe_depth(512)));



__kernel void lud_diagonal_read(__global float *m,int matrix_dim,int offset)
			 
{ 
	int tx = get_global_id(0);
	int array_offset = offset*matrix_dim+offset;
	

	write_pipe_block(p0, &m[array_offset + tx]);

}


__kernel void lud_diagonal_compute(__local  float *shadow,int   matrix_dim, int   offset){

	 int tx = get_local_id(0);float var_m;
	 int array_offset = offset*matrix_dim+offset;
	 read_pipe_block(p0, &var_m);
	int i=0;
	int j=0;
	
	for(i=0; i < BLOCK_SIZE; i++){
		shadow[i * BLOCK_SIZE + tx]=var_m;
		array_offset += matrix_dim;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
  
	for(i=0; i < BLOCK_SIZE-1; i++) {
	
	  if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    if (tx>i){
	for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }
    
	barrier(CLK_LOCAL_MEM_FENCE);
    }
	  array_offset = (offset+1)*matrix_dim+offset;
      for(i=1; i < BLOCK_SIZE; i++){
      var_m=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
      
    }
	
	write_pipe_block(p1,&var_m);

}    

    
__kernel void lud_diagonal_wb(__global float *m, int matrix_dim, int offset){

	int tx = get_local_id(0);
	int array_offset = offset*matrix_dim+offset;
	
	read_pipe_block(p1, (&m[array_offset+tx]));
    
}

