#include "bfs_fpga.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include "timer.h"
#include <stdio.h>
#include <CL/opencl.h>
#include "util.h"
#include "./util/opencl/opencl.h"

#define MAX_THREADS_PER_BLOCK 256
#define WORK_DIM 2

cl_context context = NULL;
//unsigned num_devices = 0;//////////////////////////////////////////////********************************************
void Usage(int argc, char **argv)
{

	fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
}
float eventTime(cl_event event, cl_command_queue command_queue)
{
	cl_int error = 0;
	cl_ulong eventStart, eventEnd;
	clFinish(command_queue);
	error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &eventStart, NULL);
	cl_errChk(error, "ERROR in Event Profiling.", true);
	error = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &eventEnd, NULL);
	cl_errChk(error, "ERROR in Event Profiling.", true);

	return (float)((eventEnd - eventStart) / 1e9);
}

int main(int argc, char *argv[])
{
	int no_of_nodes;
	int edge_list_size;
	int quiet = 0, platform = -1, device = -1;
	FILE *fp;
	Node *h_graph_nodes;
	char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	cl_context context = cl_init_context(platform, device, quiet);
	try
	{
		char *input_f;
		if (argc != 2)
		{
			Usage(argc, argv);
			exit(0);
		}

		input_f = argv[1];
		printf("Reading File\n");
		//Read in Graph from a file
		fp = fopen(input_f, "r");
		if (!fp)
		{
			printf("Error Reading graph file\n");
			return 0;
		}

		int source = 0;

		fscanf(fp, "%d", &no_of_nodes);

		int num_of_blocks = 1;
		int num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if (no_of_nodes > MAX_THREADS_PER_BLOCK)
		{
			num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
		}
		int work_group_size = num_of_threads_per_block;
		// allocate host memory
		h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
		h_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
		h_updating_graph_mask = (char *)malloc(sizeof(char) * no_of_nodes);
		h_graph_visited = (char *)malloc(sizeof(char) * no_of_nodes);

		int start, edgeno;
		// initalize the memory
		for (int i = 0; i < no_of_nodes; i++)
		{
			fscanf(fp, "%d %d", &start, &edgeno);
			h_graph_nodes[i].starting = start;
			h_graph_nodes[i].no_of_edges = edgeno;
			h_graph_mask[i] = false;
			h_updating_graph_mask[i] = false;
			h_graph_visited[i] = false;
		}
		//read the source node from the file
		fscanf(fp, "%d", &source);
		source = 0;
		//set the source node as true in the mask
		h_graph_mask[source] = true;
		h_graph_visited[source] = true;
		fscanf(fp, "%d", &edge_list_size);
		int id, cost;
		int *h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
		for (int i = 0; i < edge_list_size; i++)
		{
			fscanf(fp, "%d", &id);
			fscanf(fp, "%d", &cost);
			h_graph_edges[i] = id;
		}

		if (fp)
			fclose(fp);
		// allocate mem for the result on host side
		int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
		int *h_cost_ref = (int *)malloc(sizeof(int) * no_of_nodes);
		for (int i = 0; i < no_of_nodes; i++)
		{
			h_cost[i] = -1;
			h_cost_ref[i] = -1;
		}
		h_cost[source] = 0;
		h_cost_ref[source] = 0;

		//---------------------------------------------------------
		//--gpu entry
		run_bfs_gpu(context, no_of_nodes, h_graph_nodes, edge_list_size, h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);
		//---------------------------------------------------------
	}
	catch (std::string msg)
	{
		std::cout << "--cambine: exception in main ->" << msg << std::endl;
		//release host memory
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);
	}

	return 0;
}

void run_bfs_gpu(cl_context context, int no_of_nodes, Node *h_graph_nodes, int edge_list_size, int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, char *h_graph_visited, int *h_cost)
{

	// 1. set up kernel
	cl_kernel BFS_kernel_read, BFS_kernel_compute, BFS_kernel_store /*, BFS_kernel_2*/;
	cl_int status;
	float writeTime = 0, kernel_read = 0, kernel_compute = 0, kernel_store = 0, readTime = 0 /*, kernel_2 = 0*/;
	cl_program cl_BFS_program;
	cl_BFS_program = cl_compileProgram((char *)"./xclbin/krnl_bfs.hw_emu.xilinx_aws-vu9p-f1-04261818_dynamic_5_0.xclbin", NULL);

	BFS_kernel_read = clCreateKernel(cl_BFS_program, "BFS_1_Reader", &status);
	status = cl_errChk(status, (char *)"Error Creating BFS kernel_read", true);
	if (status)
		exit(1);

	BFS_kernel_compute = clCreateKernel(cl_BFS_program, "BFS_1_CU", &status);
	status = cl_errChk(status, (char *)"Error Creating BFS kernel_compute", true);
	if (status)
		exit(1);

	BFS_kernel_store = clCreateKernel(cl_BFS_program, "BFS_1_store", &status);
	status = cl_errChk(status, (char *)"Error Creating BFS kernel_store", true);
	if (status)
		exit(1);

	size_t globalWorkSize[1];
	globalWorkSize[0] = no_of_nodes;
	if (no_of_nodes % 64)
		globalWorkSize[0] += 64 - (no_of_nodes % 64);

	printf("Global Work Size: %zu\n", globalWorkSize[0]);

	////////////////////////////////////////////////////////////////////////

	char h_over;
	// 2. set up memory on device and send ipts data to device copy ipts to device
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, d_over;

	cl_int error = 0;

	d_graph_nodes = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes * sizeof(Node), h_graph_nodes, &error);
	d_graph_edges = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, edge_list_size * sizeof(int), h_graph_edges, &error);
	d_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes * sizeof(char), h_graph_mask, &error);
	d_updating_graph_mask = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes * sizeof(char), h_updating_graph_mask, &error);
	d_graph_visited = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes * sizeof(char), h_graph_visited, &error);
	d_cost = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, no_of_nodes * sizeof(int), h_cost, &error);
	d_over = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(char), &h_over, &error);

	cl_command_queue command_queue = cl_getCommandQueue();
	cl_command_queue command_queue1 = cl_getCommandQueue1();
	cl_command_queue command_queue2 = cl_getCommandQueue2();



	cl_event writeEvent, kernelEvent, kernelEvent1, kernelEvent2, /*  kernelEvent3,*/ readEvent;

	error = clEnqueueWriteBuffer(command_queue, d_graph_nodes, CL_TRUE, 0, no_of_nodes * sizeof(Node), h_graph_nodes, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer1", true);

	error = clEnqueueWriteBuffer(command_queue, d_graph_edges, CL_TRUE, 0, edge_list_size * sizeof(int), h_graph_edges, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer2", true);

	error = clEnqueueWriteBuffer(command_queue, d_graph_mask, CL_TRUE, 0, no_of_nodes * sizeof(char), h_graph_mask, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer3", true);

	error = clEnqueueWriteBuffer(command_queue, d_updating_graph_mask, CL_TRUE, 0, no_of_nodes * sizeof(char), h_updating_graph_mask, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer4", true);

	error = clEnqueueWriteBuffer(command_queue, d_graph_visited, CL_TRUE, 0, no_of_nodes * sizeof(char), h_graph_visited, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer5", true);

	error = clEnqueueWriteBuffer(command_queue, d_cost, CL_TRUE, 0, no_of_nodes * sizeof(int), h_cost, 0, NULL, &writeEvent);
	writeTime += eventTime(writeEvent, command_queue);
	clReleaseEvent(writeEvent);
	cl_errChk(error, "ERROR in clEnqueueWriteBuffer6", true);

	error = clFinish(command_queue);
	cl_errChk(error, "ERROR in command_queue", true);


	//////////////////////////////////////////////////////////////////////////////////////////////////////////READ STARTS/////

	timer kernel_timer;
	double kernel_time = 0.0;
	kernel_timer.reset();
	kernel_timer.start();
	// 3. send arguments to device
	cl_int argchk;
	do
	{
		h_over = false;
		error = clEnqueueWriteBuffer(command_queue, d_over, CL_TRUE, 0, sizeof(char), &h_over, 0, NULL, &writeEvent);
		writeTime += eventTime(writeEvent, command_queue);
		clReleaseEvent(writeEvent);
		argchk = clSetKernelArg(BFS_kernel_read, 0, sizeof(cl_mem), (void *)&d_graph_nodes);
		argchk |= clSetKernelArg(BFS_kernel_read, 1, sizeof(cl_mem), (void *)&d_graph_mask);
		cl_errChk(argchk, "ERROR in Setting BFS_read kernel args", true);

		argchk = clSetKernelArg(BFS_kernel_compute, 0, sizeof(cl_mem), (void *)&d_graph_edges);
		argchk |= clSetKernelArg(BFS_kernel_compute, 1, sizeof(cl_mem), (void *)&d_updating_graph_mask);
		argchk |= clSetKernelArg(BFS_kernel_compute, 2, sizeof(cl_mem), (void *)&d_graph_visited);
		argchk |= clSetKernelArg(BFS_kernel_compute, 3, sizeof(cl_mem), (void *)&d_cost);
		cl_errChk(argchk, "ERROR in Setting BFS_compute kernel args", true);

		// 3. send arguments to device
		argchk = clSetKernelArg(BFS_kernel_store, 0, sizeof(cl_mem), (void *)&d_graph_mask);
		cl_errChk(argchk, "ERROR in Setting BFS_WB kernel args", true);

		printf("Arguments set\n");

		// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue, BFS_kernel_read, 1, 0, globalWorkSize, NULL, 0, NULL, &kernelEvent);

		cl_errChk(error, "ERROR in Executing Kernel BFS_read", true);
		printf("Read kernel Executes\n");

		// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue1, BFS_kernel_compute, 1, 0, globalWorkSize, NULL, 0, NULL, &kernelEvent1);
		printf("Compute kernel Executes\n");
		cl_errChk(error, "ERROR in Executing Kernel BFS_compute", true);

		// 4. enqueue kernel
		error = clEnqueueNDRangeKernel(command_queue2, BFS_kernel_store, 1, 0, globalWorkSize, NULL, 0, NULL, &kernelEvent2);
		printf("Store kernel Executes\n");
		cl_errChk(error, "ERROR in Executing Kernel BFS_wb", true);

		kernel_read += eventTime(kernelEvent, command_queue);
		//~ clReleaseEvent(kernelEvent);
		kernel_compute += eventTime(kernelEvent1, command_queue1);
		//~ clReleaseEvent(kernelEvent1);
		kernel_store += eventTime(kernelEvent2, command_queue2);
		//~ clReleaseEvent(kernelEvent2);

		status = clFinish(command_queue);
		status |= clFinish(command_queue1);
		status |= clFinish(command_queue2);

		error = clEnqueueReadBuffer(command_queue2, d_cost, 1, 0, no_of_nodes * sizeof(int), h_cost, 0, NULL, &readEvent);
		cl_errChk(error, "ERROR with clEnqueueReadBuffer", true);
		readTime += eventTime(readEvent, command_queue2);
		clReleaseEvent(readEvent);

		status = clFinish(command_queue2);

	} while (h_over);

	kernel_timer.stop();
	kernel_time = kernel_timer.getTimeInSeconds();
	std::cout << "kernel time(s):" << kernel_time << std::endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////END OF OPERATION

	// 6. return finalized data and release buffers

	clReleaseMemObject(d_graph_nodes);
	clReleaseMemObject(d_graph_edges);
	clReleaseMemObject(d_graph_mask);
	clReleaseMemObject(d_updating_graph_mask);
	clReleaseMemObject(d_graph_visited);
	clReleaseMemObject(d_cost);
}
