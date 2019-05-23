 //#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

__kernel void Fan2(__global float * restrict m_dev,
                  __global float * restrict a_dev,
                  __global float * restrict b_dev,
                  const int size,
                  const int t) {	 
	 int globalIdx = get_global_id(0);
	 int globalIdy = get_global_id(1);
      if (globalIdx < size-1-t && globalIdy < size-t) {
         a_dev[size*(globalIdx+1+t)+(globalIdy+t)] -= m_dev[size*(globalIdx+1+t)+t] * a_dev[size*t+(globalIdy+t)];
		if(globalIdy == 0){
 		   b_dev[globalIdx+1+t] -= m_dev[size*(globalIdx+1+t)+(globalIdy+t)] * b_dev[t];
 	    }
 	 }
    
}
