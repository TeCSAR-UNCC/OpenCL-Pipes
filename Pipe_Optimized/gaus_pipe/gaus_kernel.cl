 
typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;



pipe float p6  __attribute__((xcl_reqd_pipe_depth(512)));
pipe float obj0_mdev_channel_00  __attribute__((xcl_reqd_pipe_depth(512)));
pipe float obj0_adev_channel_00  __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p7  __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p5 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float obj0_adev_result_channel_00  __attribute__((xcl_reqd_pipe_depth(512)));
pipe float obj0_bdev_result_channel_00  __attribute__((xcl_reqd_pipe_depth(512)));



__kernel void Fan2_Read(__global float* restrict m_dev, __global float* restrict a_dev, __global float* restrict b_dev, const int size, const int t)
{
	int globalIdx = get_global_id(0);
	int globalIdy = get_global_id(1);
	
	if(globalIdx < size-1-t && globalIdy < size-t){
		if(globalIdy == 0){
			int localbdevIdIN = globalIdx+1+t;
			int localmdevId = size*(globalIdx+1+t)+(globalIdy+t);
			
			write_pipe_block(p5, &m_dev[localmdevId]);
			write_pipe_block(p6, &b_dev[localbdevIdIN]);
		}
		
		int localXIndex = (globalIdx+1+t)* size +t;
		int localYIndex = size*t + (globalIdy+t);
		int RIndexIN = size*(globalIdx+1+t)+(globalIdy+t);
		
		write_pipe_block(obj0_mdev_channel_00, &m_dev[localXIndex]);
		write_pipe_block(obj0_adev_channel_00, &a_dev[localYIndex]);
		write_pipe_block(p7, &a_dev[RIndexIN]);
		
	}
}

__kernel void Fan2_CU(__global float* restrict b_dev, const int size, const int t)
{ 
		int globalIdx = get_global_id(0);
		int globalIdy = get_global_id(1);
		
		
		
		if(globalIdx < size-1-t && globalIdy < size-t){
			float localbdev = b_dev[t];
			float localb_dev_data, localm_dev_data, localm_dev, locala_dev, locala_dev2;
			if(globalIdy == 0){
			
			read_pipe_block(p6, &localb_dev_data);
			read_pipe_block(p5, &localm_dev_data);
			localb_dev_data -= localm_dev_data * localbdev;
			
			write_pipe_block(obj0_bdev_result_channel_00, &localb_dev_data);
			
			}
			
			read_pipe_block(obj0_mdev_channel_00, &localm_dev);
			read_pipe_block(obj0_adev_channel_00, &locala_dev);
			read_pipe_block(p7, &locala_dev2);
			locala_dev2 -= localm_dev * locala_dev;
			
			write_pipe_block(obj0_adev_result_channel_00,&locala_dev2);
		}
}

__kernel void Fan2_store(__global float* restrict a_dev, __global float* restrict b_dev, const int size, const int t)
{
		int globalIdx = get_global_id(0);
		int globalIdy = get_global_id(1);
		
		
		if(globalIdx < size-1-t && globalIdy < size-t){
			if( globalIdy == 0){
				int localbdevId = globalIdx+1+t;
				read_pipe_block(obj0_bdev_result_channel_00, &b_dev[localbdevId]);						  
			}
			int RIndex = size*(globalIdx+1+t)+(globalIdy+t);
			
			read_pipe_block(obj0_adev_result_channel_00, &a_dev[RIndex]);
		}
}


