#define BLOCK_SIZE 16


pipe float p0 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p1 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p2 __attribute__((xcl_reqd_pipe_depth(512)));


__kernel void lud_internal_read(__global float *m,int matrix_dim,int offset)
			 
{ 
	int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);
 
  
  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;
  
	int array_offset = offset*matrix_dim+offset;
	
	
	write_pipe_block(p0, &m[(offset+ty)*matrix_dim+global_col_id+tx]);
	write_pipe_block(p1, &m[(global_row_id+ty)*matrix_dim+offset+tx]);
	
}


__kernel void lud_internal_compute(__local  float *peri_row,
								   __local  float *peri_col,
								   int   matrix_dim, 
								   int   offset)
{
int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;
  float var_m1;
  float var_m2;
  float var_m3;
  read_pipe_block(p0, &var_m1);
  read_pipe_block(p1, &var_m2);
  //read_pipe_block(p1, &var_m2);
  peri_row[ty * BLOCK_SIZE + tx] = var_m1;
  peri_col[ty * BLOCK_SIZE + tx] = var_m2;
	
	barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  var_m3 -= sum;
  
	write_pipe_block(p2,&var_m3);
}    

    
__kernel void lud_internal_wb(__global float *m, int matrix_dim, int offset){

	int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;
	
	
	
	read_pipe_block(p2, (&m[(global_row_id+ty)*matrix_dim+global_col_id+tx]));
    
	
	
}

