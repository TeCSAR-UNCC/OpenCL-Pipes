/* ============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct type_{
	int starting;
	int no_of_edges;
} Node;
//--7 parameters

pipe Node foo  __attribute((xcl_reqd_pipe_depth(512)));
pipe char mask  __attribute((xcl_reqd_pipe_depth(512)));
pipe char resultmask  __attribute((xcl_reqd_pipe_depth(512)));
 
__kernel void BFS_1_Reader(const __global Node *g_graph_nodes, 
					__global char *g_graph_mask){
	int tid = get_global_id(0);



	write_pipe_block(foo, &g_graph_nodes[tid]);
	write_pipe_block(mask, &g_graph_mask[tid]);

		
}

__kernel void BFS_1_CU(const __global int *g_graph_edges,
						__global char *g_updating_graph_mask,
						__global char *g_graph_visited,
						__global int *g_cost){
	int tid = get_global_id(0);

	Node localNode;
	char localmask;
	read_pipe_block(foo, &localNode);
	read_pipe_block(mask, &localmask);    // character type 0 or 1
	//int localCost = read_pipe_block(cost);
	//printf("Kernel comp-after-read CU = %d\n", tid); 
	if(localmask){
		localmask = false;
		for( int i = localNode.starting; i< (localNode.starting + localNode.no_of_edges);i++){
			int id = g_graph_edges[i];
			if(!g_graph_visited[id]){
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
			}
		}
		
	}	
	
	write_pipe_block(resultmask, &localmask);
}

__kernel void BFS_1_store(__global char* g_graph_mask){
		
	int tid = get_global_id(0);

read_pipe_block(resultmask, g_graph_mask);

}
