#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

#define BLOCK_SIZE 256
 
pipe float p1 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p2 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p0 __attribute__((xcl_reqd_pipe_depth(512)));

__attribute__ ((reqd_work_group_size(256, 256, 1)))
__kernel 
void hotspot_read( int iteration, global float * restrict power, global float * restrict temp_src, int grid_cols, int grid_rows, int border_cols, int border_rows){
	
	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int global_id = get_global_id(0);


	// calculate the small block size
	
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	
	// calculate the boundary for the block according to 
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;

	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;
	
	// load data if it is within the valid input range
	int index = grid_cols*yidx+xidx;
	
	if(IN_RANGE(yidx, 0, grid_rows-1) && IN_RANGE(xidx, 0, grid_cols-1)){
		
		
		write_pipe_block(p0, &temp_src[index]);
		write_pipe_block(p1, &power[index]);
	}


}

__kernel __attribute__ ((reqd_work_group_size(256, 256, 1)))
void hotspot_CU(int iteration, int grid_cols, int grid_rows, int border_cols, int border_rows, float Cap, float Rx, float Ry, float Rz, float step) {
	float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temporary temperature result
	
	float amb_temp = 80.0f;
	float step_div_Cap;
	float Rx_1,Ry_1,Rz_1;

	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	
	step_div_Cap=step/Cap;

	Rx_1=1/Rx;
	Ry_1=1/Ry;
	Rz_1=1/Rz;
	
	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//
	// calculate the boundary for the block according to 
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BLOCK_SIZE-1;
	int blkXmax = blkX+BLOCK_SIZE-1;
	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;
	if(IN_RANGE(yidx, 0, grid_rows-1) && IN_RANGE(xidx, 0, grid_cols-1)){
		
		read_pipe_block(p0, &temp_on_cuda[ty][tx]); 
		read_pipe_block(p1, &power_on_cuda[ty][tx]); 

	}
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;
	
	int N = ty-1;
	int S = ty+1;
	int W = tx-1;
	int E = tx+1;
	
	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;
	
	bool computed;

	for (int i=0; i<iteration ; i++)
	{ 	
		computed = false;
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(tx, validXmin, validXmax) && \
		IN_RANGE(ty, validYmin, validYmax) ) {

			computed = true;
			temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
			(temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 + 
			(temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 + 
			(amb_temp - temp_on_cuda[ty][tx]) * Rz_1);


		}

		if(i==iteration-1)
			break;
		if(computed)	
		{temp_on_cuda[ty][tx]= temp_t[ty][tx];
		}
	}

	if(computed)	
	{
		write_pipe_block(p2, &temp_t[ty][tx]);
	}
}

__kernel __attribute__ ((reqd_work_group_size(256, 256, 1)))
void hotspot_store(int iteration, global float * restrict temp_dst, int grid_cols, int grid_rows, int border_cols, int border_rows){
	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);
	float temp_tmp[BLOCK_SIZE][BLOCK_SIZE];

	int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE
	
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BLOCK_SIZE-1;
	int blkXmax = blkX+BLOCK_SIZE-1;
	int yidx = blkY+ty;
	int xidx = blkX+tx;
	
	int index = grid_cols*yidx+xidx;
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;
	bool compute;
		for (int i=0; i<iteration ; i++)
		{  compute = false; 
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
		IN_RANGE(tx, validXmin, validXmax) && \
		IN_RANGE(ty, validYmin, validYmax) ) {
		compute = true;

        }
        	if(i==iteration-1)
			break;

	}
	if(compute)
	 { 	
    	read_pipe_block(p2, &temp_dst[index]);
    }
}

