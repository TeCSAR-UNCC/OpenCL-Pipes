

typedef struct latLong
    {
        float lat;
        float lng;
    } LatLong;

pipe LatLong p0 __attribute__((xcl_reqd_pipe_depth(512)));
pipe float p1 __attribute__((xcl_reqd_pipe_depth(512)));


__kernel void NearestNeighbor_read(__global LatLong *d_locations,const int numRecords ) {

	int globalId = get_global_id(0);
	write_pipe_block(p0, &d_locations[globalId]);
}



__kernel void NearestNeighbor( const int numRecords,
							  const float lat,
							  const float lng) {
	 int globalId = get_global_id(0);
	 LatLong loc_lat;
	 read_pipe_block(p0, &loc_lat);						  
     
		    float d_dist = (lat-loc_lat.lat)*(lat-loc_lat.lat)+(lng-loc_lat.lng)*(lng-loc_lat.lng);
			float d_distances = (float)sqrt(d_dist);
			write_pipe_block(p1,&d_distances);			
}

__kernel void NearestNeighbor_wb(__global float * d_distances, const int numRecords) {

	int globalId = get_global_id(0);
	read_pipe_block(p1, (d_distances+globalId)); 

}
