#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include "freeglut.h"
#include <gl\gl.h>                                // Header File For The OpenGL32 Library
#include <gl\glu.h>                               // Header File For The GLu32 Library
#include <float.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>

#define M_PI       3.14159265358979323846
#define EPSILON	   0.000001

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static bool key_down[256];
static bool dir[4];

static float3 *h_tri_list1 = NULL;
static float3 *h_tri_list2 = NULL;
static float3 *d_tri_list1 = NULL;
static float3 *d_tri_list2 = NULL;

static int tri_count1 = 0;
static int tri_count2 = 0;

static float3 list1_pos;
static float3 list2_pos;

static float orientation1[16];
static float orientation2[16];

static float user_control[16];
static float3 *forward = (float3 *) &user_control[8];
static float3 *up = (float3 *) &user_control[4];
static float3 *left = (float3 *) &user_control[0];
float3 cam_pos;
int xDifference = 0;
int yDifference = 0; 
bool use_cuda = false;
bool use_cpu = false;
bool draw_line = false;
bool draw_cuda_line = false;

float3 x1, x2;

int frame=0,timed = 0,timebase=0;

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

//
//		float3 shortcuts
//

__device__ __host__ float3 f3_add(float3 A, float3 B)
{
	float3 res;

	res.x = A.x + B.x;
	res.y = A.y + B.y;
	res.z = A.z + B.z;

	return res;
}

__device__ __host__ float3 f3_sub(float3 A, float3 B)
{
	float3 res;

	res.x = A.x - B.x;
	res.y = A.y - B.y;
	res.z = A.z - B.z;

	return res;
}

__device__ __host__ float f3_dot(float3 A, float3 B)
{
	float res;

	res = A.x * B.x + A.y * B.y + A.z * B.z;

	return res;
}

__device__ __host__ float3 f3_crss(float3 A, float3 B)
{
	float3 res;

	res.x = A.y * B.z - A.z * B.y;
	res.y = A.z * B.x - A.x - B.z;
	res.z = A.x * B.y - A.y - B.x;

	return res;
}

__device__ __host__ float3 f3_sclrmult(float val, float3 A)
{
	float3 res;

	res.x = val * A.x;
	res.y = val * A.y;
	res.z = val * A.z;

	return res;
}

__device__ __host__ float f_clamp(float n, float min, float max)
{
	if (n < min) return min;
	if (n > max) return max;
	return n;
}

__device__ __host__ float3 f3_transform(float3 vtx, float mdl[])
{
	float3 res;

	res.x = mdl[0] * vtx.x + mdl[4] * vtx.y + mdl[8] * vtx.z + mdl[12];
	res.y = mdl[1] * vtx.x + mdl[5] * vtx.y + mdl[9] * vtx.z + mdl[13];
	res.z = mdl[2] * vtx.x + mdl[6] * vtx.y + mdl[10] * vtx.z + mdl[14];

	return res;
}

//
//		closest distance functions 
//

__device__ __host__ float point_to_triangle(float3 pt, float3 &ptt, float3 a, float3 b, float3 c)
{
	// Check if P in vertex region outside A
	float3 ab = f3_sub(b, a);
	float3 ac = f3_sub(c, a);
	float3 ap = f3_sub(pt, a);

	float d1 = f3_dot(ab, ap);
	float d2 = f3_dot(ac, ap);

	if(d1 <= 0.0f && d2 <= 0.0f) 
	{
		ptt = a;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1,0,0)
	}
	// Check if P in vertex region outside B
	float3 bp = f3_sub(pt, b);
	
	float d3 = f3_dot(ab, bp);
	float d4 = f3_dot(ac, bp);

	if(d3 >= 0.0f && d4 <= d3) 
	{
		ptt = b;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (0,1,0)
	}
	// Check if P in edge region of AB, if so return projection of P onto AB
	float vc = d1*d4 - d3*d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) 
	{
		float v = d1 / (d1 - d3);
		ptt = f3_add(a, f3_sclrmult(v, ab)); 
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1-v,v,0)
	}

	// Check if P in vertex region outside C
	float3 cp = f3_sub(pt, c);
	float d5 = f3_dot(ab, cp);
	float d6 = f3_dot(ac, cp);

	if(d6 >= 0.0f && d5 <= d6) // barycentric coordinates (0,0,1)
	{
		ptt = c;
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); 
	}
	// Check if P in edge region of AC, if so return projection of P onto AC
	float vb = d5*d2 - d1*d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float w = d2 / (d2 - d6);
		ptt = f3_add(a, f3_sclrmult(w, ac));
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (1-w,0,w)
	}

	// Check if P in edge region of BC, if so return projection of P onto BC
	float va = d3*d6 - d5*d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) 
	{
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		ptt = f3_add(b, f3_sclrmult(w, f3_sub(c, b)));
		float3 dist = f3_sub(pt, ptt);
		return f3_dot(dist, dist); // barycentric coordinates (0,1-w,w)
	}

	// P inside face region. Compute Q through its barycentric coordinates (u,v,w)
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	ptt = f3_add(a, f3_add(f3_sclrmult(v, ab), f3_sclrmult(w, ac))); // = u*a + v*b + w*c, u = va * denom = 1.0f - v - w
	float3 dist = f3_sub(pt, ptt);
	return f3_dot(dist, dist);
}

__device__ __host__ float edge_to_edge(float3 p1, float3 q1, float3 p2, float3 q2,
									   float &s, float &t, float3 &c1, float3 &c2)
{
	float3 d1 = f3_sub(q1, p1); // Direction vector of segment S1
	float3 d2 = f3_sub(q2, p2); // Direction vector of segment S2
	float3 r = f3_sub(p1, p2);

	float a = f3_dot(d1, d1); // Squared length of segment S1, always nonnegative
	float e = f3_dot(d2, d2); // Squared length of segment S2, always nonnegative
	float f = f3_dot(d2, r);

	// Check if either or both segments degenerate into points
	if (a <= EPSILON && e <= EPSILON) 
	{
		// Both segments degenerate into points
		s = t = 0.0f;
		c1 = p1;
		c2 = p2;
		float3 c2c1 = f3_sub(c1, c2);
		return f3_dot(c2c1, c2c1);
	}

	if (a <= EPSILON) 
	{
		// First segment degenerates into a point
		s = 0.0f;
		t = f / e; // s = 0 => t = (b*s + f) / e = f / e
		t = f_clamp(t, 0.0f, 1.0f);
	} 
	else 
	{
		float c = f3_dot(d1, r);
		if (e <= EPSILON) 
		{
			// Second segment degenerates into a point
			t = 0.0f;
			s = f_clamp(-c / a, 0.0f, 1.0f); // t = 0 => s = (b*t - c) / a = -c / a
		}
		else 
		{
			// The general nondegenerate case starts here
			float b = f3_dot(d1, d2);
			float denom = a*e - b*b; // Always nonnegative
			// If segments not parallel, compute closest point on L1 to L2 and
			// clamp to segment S1. Else pick arbitrary s (here 0)
			if (denom != 0.0f) 
			{
				s = f_clamp((b*f - c*e) / denom, 0.0f, 1.0f);
			} 
			else 
				s = 0.0f;
			// Compute point on L2 closest to S1(s) using
			// t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e

			// If t in [0,1] done. Else clamp t, recompute s for the new value
			// of t using s = Dot((P2 + D2*t) - P1,D1) / Dot(D1,D1)= (t*b - c) / a
			// and clamp s to [0, 1]
			float tnom = b*s + f;
			if (tnom < 0.0f) 
			{
				t = 0.0f;
				s = f_clamp(-c / a, 0.0f, 1.0f);
			} 
			else if (tnom > e) 
			{
				t = 1.0f;
				s = f_clamp((b - c) / a, 0.0f, 1.0f);
			}
			else
			{
				t = tnom / e;
			}
		}
	}

	c1 = f3_add(p1, f3_sclrmult(s, d1));
	c2 = f3_add(p2, f3_sclrmult(t, d2));
	float3 c2c1 = f3_sub(c1, c2);
	return f3_dot(c2c1, c2c1);
}

__device__ __host__ float triangle_triangle(float3 a1, float3 b1, float3 c1,
										    float3 a2, float3 b2, float3 c2,
										    float3 &pr1, float3 &pr2)
{
	float local_min;
	float temp;
	float s = 0, t = 0;
	float3 p1;
	float3 p2;

	local_min = edge_to_edge(a1, b1, a2, b2, s, t, p1, p2);
	pr1 = p1;
	pr2 = p2;
	temp = edge_to_edge(a1, c1, a2, b2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, a2, b2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, b1, a2, c2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, c1, a2, c2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, a2, c2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, b1, c2, b2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(a1, c1, c2, b2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	temp = edge_to_edge(c1, b1, c2, b2, s, t, p1, p2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = a1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = b1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p1 = c1;
	temp = point_to_triangle(p1, p2, a2, b2, c2);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = a2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = b2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}
	p2 = c2;
	temp = point_to_triangle(p2, p1, a1, b1, c1);
	if(temp < local_min)
	{
		local_min = temp;
		pr1 = p1;
		pr2 = p2;
	}

	float3 dist = f3_sub(pr1, pr2);
	return f3_dot(dist, dist);
}

float list_list(float3 *list1, float3 *list2, int size1, int size2, float3 &pr1, float3 &pr2)
{
	float local_min = FLT_MAX;
	float3 p1, p2;
	float temp;

	for(int i = 0; i < size1; i++)
	{
		for(int j = 0; j < size2; j++)
		{
			temp = triangle_triangle(list1[i], list1[i+size1], list1[i+size1*2],
									 list2[j], list2[j+size2], list2[j+size2*2],
									 p1, p2);
			if(temp < local_min)
			{
				local_min = temp;
				pr1 = p1;
				pr2 = p2;
			}
		}
	}

	float3 dist = f3_sub(pr1, pr2);
	return f3_dot(dist, dist);
}


__global__ void cuda_list_list(float3 *list1, float3 *list2, int size1, int size2, float* dist, int *i_index)
{
	unsigned int x = (blockIdx.x*blockDim.x + threadIdx.x);
    unsigned int y = (blockIdx.y*blockDim.y + threadIdx.y);

	if(x < size1)
	{
		if(y < size2)
		{
			float3 pr1, pr2;
			i_index[x * size2 + y] = x * size2 + y;
			dist[x * size2 + y] = triangle_triangle(list1[x], list1[x+size1], list1[x+size1*2],
													list2[y], list2[y+size2], list2[y+size2*2],
													pr1, pr2);
		}
	}

}

template <unsigned int blockSize, bool nIsPow2>
__global__ void find_min_reduce(float *i_list, float *o_list, int *i_index, int *o_index, int n, float f_max)
{
	extern __shared__ float s_data[];

    float* s_list = (float*)s_data;
	int* s_index = (int*)&s_list[blockSize];
	

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float my_min = f_max;
	int my_index;
	float temp;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
		temp = i_list[i];
		if(temp < my_min)
		{
			my_min = temp;
			my_index = i_index[i];
		}
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
		{
			temp = i_list[i+blockSize];
			if(temp < my_min)
			{
				my_min = temp;
				my_index = i_index[i+blockSize];
			}
		}
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    s_list[tid] = my_min;
	s_index[tid] = my_index;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) 
	{
		if (tid < 256) 
		{
			temp = s_list[tid + 256];
			if(temp < my_min)
			{
				my_min = temp;
				s_list[tid] = my_min;
				s_index[tid] = s_index[tid + 256];
			}
		} 
		__syncthreads(); 
	}
    if (blockSize >= 256) 
	{ 
		if (tid < 128) 
		{
			temp = s_list[tid + 128];
			if(temp < my_min)
			{
				my_min = temp;
				s_list[tid] = my_min;
				s_index[tid] = s_index[tid + 128];
			}
		}
		__syncthreads(); 
	}
	if (blockSize >= 128) 
	{
		if (tid <  64)
		{
			temp = s_list[tid + 64];
			if(temp < my_min)
			{
				my_min = temp;
				s_list[tid] = my_min;
				s_index[tid] = s_index[tid + 64];
			}
		}
		__syncthreads(); 
	}
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float* smeml = s_list;
		volatile int* smemi = s_index;

        if (blockSize >=  64) 
		{
			temp = smeml[tid + 32];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 32];
			}
		}
        if (blockSize >=  32)
		{
			temp = smeml[tid + 16];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 16];
			}		}
        if (blockSize >=  16) 
		{
			temp = smeml[tid + 8];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 8];
			}	
		}
        if (blockSize >=   8)
		{
			temp = smeml[tid + 4];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 4];
			}	
		}
        if (blockSize >=   4)
		{
			temp = smeml[tid + 2];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 2];
			}		
		}
        if (blockSize >=   2)
		{
			temp = smeml[tid + 1];
			if(temp < my_min)
			{
				my_min = temp;
				smeml[tid] = my_min;
				smemi[tid] = smemi[tid + 1];
			}		
		}
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
	{
		o_list[blockIdx.x] = s_list[0];
        o_index[blockIdx.x] = s_index[0];
	}
}

void reduce(int size, int threads, int blocks, float *d_ilist, int *d_iindex, float *d_olist, int *d_oindex)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = 2 * ((threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float));

	if (isPow2(size))
    {
        switch (threads)
        {
		case 512:
			find_min_reduce <512, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 256:
			find_min_reduce <256, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 128:
			find_min_reduce <128, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 64:
			find_min_reduce <64, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 32:
			find_min_reduce <32, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 16:
			find_min_reduce <16, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 8:
			find_min_reduce <8, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 4:
			find_min_reduce <4, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 2:
			find_min_reduce <2, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 1:
			find_min_reduce <1, true> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
        }
    }
    else
    {
        switch (threads)
        {
		case 512:
			find_min_reduce <512, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 256:
			find_min_reduce <256, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 128:
			find_min_reduce <128, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 64:
			find_min_reduce <64, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 32:
			find_min_reduce <32, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 16:
			find_min_reduce <16, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 8:
			find_min_reduce <8, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 4:
			find_min_reduce <4, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 2:
			find_min_reduce <2, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
		case 1:
			find_min_reduce <1, false> <<< dimGrid, dimBlock, smemSize >>> (d_ilist, d_olist, d_iindex, d_oindex, size, FLT_MAX);
			break;
        }
    }
}

void launch_kernel(float3 *d_ilist1, float3 *d_ilist2, int lsize1, int lsize2, float3 &pr1, float3 &pr2, int maxBlocks, int maxThreads)
{
	cudaError_t myErrorFlag;

	int n = lsize1 * lsize2;

	dim3 block(8, 8, 1);

	int grid1 = tri_count1 / block.x + (tri_count1 % block.x == 0 ? 0:1);
	int grid2 = tri_count2 / block.y + (tri_count2 % block.y == 0 ? 0:1);
    dim3 grid(grid1, grid2, 1);

	float *d_dist, *d_odist;
	int *d_index, *d_oindex;
	
	myErrorFlag = cudaMalloc((void **) &d_dist, sizeof(float) * n );
	myErrorFlag = cudaMalloc((void **) &d_index, sizeof(int) * n );

	cuda_list_list <<< grid, block >>> (d_ilist1, d_ilist2, lsize1, lsize2, d_dist, d_index);
	myErrorFlag = cudaThreadSynchronize();

	//float *hlist;
	//int *hindex;
	//hlist = (float*)malloc(sizeof(float) * n);
	//hindex = (int*)malloc(sizeof(int) * n);
	//myErrorFlag = cudaMemcpy( hlist, d_dist, sizeof(float) * n, cudaMemcpyDeviceToHost);
	//myErrorFlag = cudaMemcpy( hindex, d_index, sizeof(float) * n, cudaMemcpyDeviceToHost);

	
	int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = min(maxBlocks, blocks);

	myErrorFlag = cudaMalloc((void **) &d_odist, sizeof(float) * blocks );
	myErrorFlag = cudaMalloc((void **) &d_oindex, sizeof(int) * blocks );

	reduce(n, threads, blocks, d_dist, d_index, d_odist, d_oindex);
	myErrorFlag = cudaThreadSynchronize();
	//myErrorFlag = cudaMemcpy( hlist, d_odist, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
	//myErrorFlag = cudaMemcpy( hindex, d_oindex, sizeof(int) * blocks, cudaMemcpyDeviceToHost);
	int s = blocks;
	while(s > 1) 
	{
		threads = (s < maxThreads*2) ? nextPow2((s + 1)/ 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);
		blocks = min(maxBlocks, blocks);

		reduce(s, threads, blocks, d_odist, d_oindex, d_odist, d_oindex);
		myErrorFlag = cudaThreadSynchronize();
		//myErrorFlag = cudaMemcpy( hlist, d_odist, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
		//myErrorFlag = cudaMemcpy( hindex, d_oindex, sizeof(float) * blocks, cudaMemcpyDeviceToHost);
		s = (s + (threads*2-1)) / (threads*2);
	}
	myErrorFlag = cudaThreadSynchronize();

	int result_ind;
	myErrorFlag = cudaMemcpy( &result_ind, &d_oindex[0], sizeof(int), cudaMemcpyDeviceToHost);

	//triangle indexes
	int x = result_ind / lsize2;
	int y = result_ind - x * lsize2;

	float dist = triangle_triangle(h_tri_list1[x], h_tri_list1[x + tri_count1], h_tri_list1[x + tri_count1 * 2],
								   h_tri_list2[y], h_tri_list2[y + tri_count2], h_tri_list2[y + tri_count2 * 2],
								   pr1, pr2);

	//myErrorFlag = cudaMemcpy( &pr1, &d_ilist1[x], sizeof(float3), cudaMemcpyDeviceToHost);
	//myErrorFlag = cudaMemcpy( &pr2, &d_ilist2[y], sizeof(float3), cudaMemcpyDeviceToHost);
	cudaFree(d_dist);
	cudaFree(d_index);
	cudaFree(d_odist);
	cudaFree(d_oindex);
}


static void clean_up()
{
	free(h_tri_list1);
	free(h_tri_list2);
	cudaFree(d_tri_list1);
	cudaFree(d_tri_list2);
}

int get_line(FILE *file, char line[], int max)
{
	int nch = 0;
	int c;
	max = max - 1;			/* leave room for '\0' */

	while((c = fgetc(file)) != EOF)
	{
		if(c == '\n')
			break;

		if(nch < max)
		{
			line[nch] = c;
			nch = nch + 1;
		}
	}

	if(c == EOF && nch == 0)
		return EOF;

	line[nch] = '\0';
	return nch;
}

static void process_input()
{
	if(key_down[8])
	{
		cam_pos.x = 0;
		cam_pos.y = 0;
		cam_pos.z = 0;
		glPushMatrix();
		glMatrixMode (GL_MODELVIEW);	
		glLoadIdentity();
		glGetFloatv(GL_MODELVIEW_MATRIX, user_control);
		glPopMatrix();
		xDifference = 0;
		yDifference = 0;
	}
	if(key_down[120])
	{
		cam_pos.x += up->x;
		cam_pos.y += up->y;
		cam_pos.z += up->z;
	}
	if(key_down[122])
	{
		cam_pos.x -= up->x;
		cam_pos.y -= up->y;
		cam_pos.z -= up->z;
	}
	if(key_down[119])
	{
		cam_pos.x += forward->x;
		cam_pos.y += forward->y;
		cam_pos.z += forward->z;
	}
	if(key_down[115])
	{
		cam_pos.x -= forward->x;
		cam_pos.y -= forward->y;
		cam_pos.z -= forward->z;
	}
	if(key_down[97])
	{
		cam_pos.x += left->x;
		cam_pos.y += left->y;
		cam_pos.z += left->z;
	}
	if(key_down[100])
	{
		cam_pos.x -= left->x;
		cam_pos.y -= left->y;
		cam_pos.z -= left->z;
	}
	if(key_down[113])
	{
		exit(0);
	}
	if(dir[0])
	{
		xDifference += 50;
	}
	if(dir[1])
	{
		yDifference -= 50;
	}
	if(dir[2])
	{
		xDifference -= 50;
	}
	if(dir[3])
	{
		yDifference += 50;
	}	
}

static void draw_stl(float3 *obj, int size)
{
	if(size > 0);
	{
		glBegin(GL_TRIANGLES);
		for(int i = 0; i < size; i++)
		{
			glVertex3f(obj[i].x, obj[i].y, obj[i].z);
			glVertex3f(obj[i + size].x, obj[i + size].y, obj[i + size].z);
			glVertex3f(obj[i + size * 2].x, obj[i + size * 2].y, obj[i + size * 2].z);
		}
		glEnd();
	}
}

static void update_mesh(float &update, float &transfer)
{
	glMatrixMode (GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glGetFloatv(GL_MODELVIEW_MATRIX, user_control);
	glRotatef(10, 1, 0, 1);
	glGetFloatv(GL_MODELVIEW_MATRIX, orientation1);
	glRotatef(-15, 0, 1, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, orientation2);
	glPopMatrix();


	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER stop;

	QueryPerformanceCounter(&start);

	for(int i = 0; i < tri_count1 * 3; i++)
	{
		h_tri_list1[i] = f3_transform(h_tri_list1[i], orientation1);
	}

	for(int i = 0; i < tri_count2 * 3; i++)
	{
		h_tri_list2[i] = f3_transform(h_tri_list2[i], orientation2);
	}

	QueryPerformanceCounter(&stop);
	QueryPerformanceFrequency(&frequency);

	double qtime = ((double)(stop.QuadPart - start.QuadPart)) / ((double)(frequency.QuadPart));
	printf("(%d triangles)Meshes updated in: %f ms.\n", tri_count1 + tri_count2, qtime * 1000);


	cudaEvent_t cstart, cstop;
	float et;
	cudaError_t myErrorFlag = cudaEventCreate(&cstart); 
	myErrorFlag = cudaEventCreate(&cstop);
	myErrorFlag = cudaEventRecord(cstart, 0);

	myErrorFlag = cudaMemcpy(d_tri_list1, h_tri_list1, tri_count1 * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	myErrorFlag = cudaMemcpy(d_tri_list2, h_tri_list2, tri_count2 * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	
	myErrorFlag = cudaEventRecord(cstop, 0);
	myErrorFlag = cudaEventSynchronize(cstop);

	if(myErrorFlag == cudaSuccess);
	{
		cudaEventElapsedTime(&et, cstart, cstop);

		printf("Meshes transfered to GPU in: %f ms. \n", et);
	}
	draw_line = false;

	update = (float)(qtime * 1000);
	transfer = et;


}

static void run_test()
{
	float total_up = 0;
	float total_tra = 0;
	float total_com = 0;

	for(int i = 0; i < 100; i++)
	{
		float upd, tra;
		update_mesh(upd, tra);
		total_up += upd;
		total_tra += tra;

		float et;
		float timeGPU = 0;
		cudaError_t myError;
		cudaEvent_t start, stop;
		myError = cudaEventCreate(&start); 
		myError = cudaEventCreate(&stop);
		myError = cudaEventRecord(start, 0);

		launch_kernel(d_tri_list1, d_tri_list2, tri_count1, tri_count2, x1, x2, 64, 128);
		float3 dist = f3_sub(x2, x1);

		myError = cudaEventRecord(stop, 0);
		myError = cudaEventSynchronize(stop);
				
		if(myError == cudaSuccess);
		{
			cudaEventElapsedTime(&et, start, stop);

			printf("Lists compared with GPU in: %f ms. \n min squared distance between meshes: %f \n\n", et, f3_dot(dist, dist));
			total_com += et;
			draw_line = true;
			draw_cuda_line = true;
		}

		glClearColor (0.0,0.0,0.0,1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glLoadIdentity();  
		gluLookAt (cam_pos.x, cam_pos.y, cam_pos.z,
			cam_pos.x + forward->x, cam_pos.y + forward->y, cam_pos.z + forward->z,
			up->x, up->y, up->z);



		//glEnable(GL_LIGHTING);
		//glEnable(GL_LIGHT0);
		glPushMatrix();

		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		glColor3f(1.0f, 1.0f, 1.0f);
		draw_stl(h_tri_list1, tri_count1);

		//glPushMatrix();
		//glTranslatef(-8, 0, 0);
		//glScalef(-1, 1, 1);
		//draw_stl(h_tri_list1, tri_count1);
		//glPopMatrix();

		draw_stl(h_tri_list2, tri_count2);

		if(draw_line)
		{
			glColor3f(1, 0, 0);
			if(draw_cuda_line)
			{
				glColor3f(0, 1, 0);
			}
			glBegin(GL_LINES);
			glVertex3f(x1.x, x1.y, x1.z);
			glVertex3f(x2.x, x2.y, x2.z);
			glEnd();
		}


		glPopMatrix();

		//glDisable(GL_LIGHT0);
		//glDisable(GL_LIGHTING);

		glutSwapBuffers();
		timebase = timed;
		timed = glutGet(GLUT_ELAPSED_TIME);
		//printf("%d \n", timed-timebase);
		
	}

	printf("Avarage update time: %f ms. \n", total_up / 100);
	printf("Avarage transfer time: %f ms. \n", total_tra / 100);
	printf("Avarage computation time: %f ms. \n", total_com / 100);

}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func_down ( unsigned char key, int x, int y )
{
	key_down[key] = true;
}
static void key_func_up ( unsigned char key, int x, int y )
{
	key_down[key] = false;
}

static void special_key_down ( int key, int x, int y )
{
	switch(key)
	{
		case GLUT_KEY_LEFT:
			dir[0] = true;
			break;
		case GLUT_KEY_UP:
			dir[1] = true;
			break;
		case GLUT_KEY_RIGHT:
			dir[2] = true;
			break;
		case GLUT_KEY_DOWN:
			dir[3] = true;
			break;
	}
}
static void special_key_up ( int key, int x, int y )
{
	switch(key)
	{
		case GLUT_KEY_LEFT:
			dir[0] = false;
			break;
		case GLUT_KEY_UP:
			dir[1] = false;
			break;
		case GLUT_KEY_RIGHT:
			dir[2] = false;
			break;
		case GLUT_KEY_DOWN:
			dir[3] = false;
			break;
	}
}

static void mouse_func ( int button, int state, int x, int y )
{

}

static void motion_func ( int x, int y )
{

}

static void reshape_func ( int width, int height )
{
	win_x = width;
	win_y = height;
	glViewport (0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (60, (GLfloat)width / (GLfloat)height, 0.1f, 1000.0f);
    glMatrixMode (GL_MODELVIEW);
}

static void idle_func ( void )
{
	//frame++;
	//timed=glutGet(GLUT_ELAPSED_TIME);

	//if (timed - timebase > 1000) {
	//	int fps = frame*1000.0/(timed-timebase);
	// 	timebase = timed;
	//	frame = 0;
	//	printf("%d \n", fps);
	//}
	process_input();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(((float)xDifference*M_PI)/180.0f, 0, 1, 0);
	glRotatef(((float)yDifference*M_PI)/180.0f, 1, 0, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, user_control);
	glPopMatrix();
	glutPostRedisplay ();

	if(use_cpu)
	{
		LARGE_INTEGER frequency;
		LARGE_INTEGER start;
		LARGE_INTEGER stop;

		QueryPerformanceCounter(&start);

	
		float dist;
		dist = list_list(h_tri_list1, h_tri_list2, tri_count1, tri_count2, x1, x2);

		QueryPerformanceCounter(&stop);
		QueryPerformanceFrequency(&frequency);

		double qtime = ((double)(stop.QuadPart - start.QuadPart)) / ((double)(frequency.QuadPart));
		printf("Lists compared with CPU in: %f ms.\n min squared distance between meshes: %f \n\n", qtime * 1000, dist);
		use_cpu = false;
		draw_line = true;
		draw_cuda_line = false;
	}
	if(use_cuda)
	{
		//launch_kernel

		float et;
		float timeGPU = 0;
		cudaError_t myError;
		cudaEvent_t start, stop;
		myError = cudaEventCreate(&start); 
		myError = cudaEventCreate(&stop);
		myError = cudaEventRecord(start, 0);

		launch_kernel(d_tri_list1, d_tri_list2, tri_count1, tri_count2, x1, x2, 64, 128);
		float3 dist = f3_sub(x2, x1);

		myError = cudaEventRecord(stop, 0);
		myError = cudaEventSynchronize(stop);
				
		if(myError == cudaSuccess);
		{
			cudaEventElapsedTime(&et, start, stop);

			printf("Lists compared with GPU in: %f ms. \n min squared distance between meshes: %f \n\n", et, f3_dot(dist, dist));
			
			draw_line = true;
			draw_cuda_line = true;
		}
		use_cuda = false;
	}

}

static void display_func ( void )
{

	glClearColor (0.0,0.0,0.0,1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();  
	gluLookAt (cam_pos.x, cam_pos.y, cam_pos.z,
			   cam_pos.x + forward->x, cam_pos.y + forward->y, cam_pos.z + forward->z,
			   up->x, up->y, up->z);



	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	glPushMatrix();
	
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glColor3f(1.0f, 1.0f, 1.0f);
	draw_stl(h_tri_list1, tri_count1);

	//glPushMatrix();
	//glTranslatef(-8, 0, 0);
	//glScalef(-1, 1, 1);
	//draw_stl(h_tri_list1, tri_count1);
	//glPopMatrix();

	draw_stl(h_tri_list2, tri_count2);

	if(draw_line)
	{
		glColor3f(1, 0, 0);
		if(draw_cuda_line)
		{
			glColor3f(0, 1, 0);
		}
		glBegin(GL_LINES);
		glVertex3f(x1.x, x1.y, x1.z);
		glVertex3f(x2.x, x2.y, x2.z);
		glEnd();
	}
	

	glPopMatrix();

	//glDisable(GL_LIGHT0);
	//glDisable(GL_LIGHTING);

	glutSwapBuffers();
	timebase = timed;
	timed = glutGet(GLUT_ELAPSED_TIME);
	//printf("%d \n", timed-timebase);

	
}

static void main_menu_func ( int i)
{
	if(i == 0)
		use_cpu = true;
	if(i == 1)
		use_cuda = true;
	if(i == 2)
	{
		float upd, tra;
		update_mesh(upd, tra);
	}
	if(i == 3)
		run_test();

}

static void open_glut_window ( void )
{
	cudaError_t myErrorFlag;
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( win_x, win_y );
	win_id = glutCreateWindow ( "Right Click for Main Menu" );
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	glutSwapBuffers ();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSwapBuffers ();

	GLfloat ambient[] = { 0.2, 0.2, 0.2, 0.0 };
    GLfloat diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
    GLfloat position[] = { 0.0, 0.0, 50.0, 0.0 };
    GLfloat lmodel_ambient[] = { 0.2, 0.2, 0.2, 0.0 };

	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

	
	glMatrixMode (GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glGetFloatv(GL_MODELVIEW_MATRIX, user_control);
	//glRotatef(90, 0, 0, 1);
	glGetFloatv(GL_MODELVIEW_MATRIX, orientation1);
	//glRotatef(-180, 0, 0, 1);
	glTranslatef(-5, 0, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, orientation2);
	glPopMatrix();


	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER stop;

	QueryPerformanceCounter(&start);

	for(int i = 0; i < tri_count1 * 3; i++)
	{
		h_tri_list1[i] = f3_transform(h_tri_list1[i], orientation1);
	}

	for(int i = 0; i < tri_count2 * 3; i++)
	{
		h_tri_list2[i] = f3_transform(h_tri_list2[i], orientation2);
	}

	QueryPerformanceCounter(&stop);
	QueryPerformanceFrequency(&frequency);

	double qtime = ((double)(stop.QuadPart - start.QuadPart)) / ((double)(frequency.QuadPart));
	printf("(%d triangles)Meshes updated in: %f ms.\n", tri_count1 + tri_count2, qtime * 1000);


	cudaEvent_t cstart, cstop;
	float et;
	myErrorFlag = cudaEventCreate(&cstart); 
	myErrorFlag = cudaEventCreate(&cstop);
	myErrorFlag = cudaEventRecord(cstart, 0);

	myErrorFlag = cudaMalloc((void **) &d_tri_list1, tri_count1 * 3 * sizeof(float3));   
	myErrorFlag = cudaMalloc((void **) &d_tri_list2, tri_count2 * 3 * sizeof(float3));   
	myErrorFlag = cudaMemcpy(d_tri_list1, h_tri_list1, tri_count1 * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	myErrorFlag = cudaMemcpy(d_tri_list2, h_tri_list2, tri_count2 * 3 * sizeof(float3), cudaMemcpyHostToDevice);
	
	myErrorFlag = cudaEventRecord(cstop, 0);
	myErrorFlag = cudaEventSynchronize(cstop);

	if(myErrorFlag == cudaSuccess);
	{
		cudaEventElapsedTime(&et, cstart, cstop);

		printf("Meshes transfered to GPU in: %f ms. \n", et);
	}


	glutKeyboardUpFunc ( key_func_up );
	glutKeyboardFunc ( key_func_down );
	glutSpecialFunc (special_key_down);
	glutSpecialUpFunc (special_key_up);
	glutMouseFunc ( mouse_func );
	glutMotionFunc ( motion_func );
	glutReshapeFunc ( reshape_func );
	glutIdleFunc ( idle_func );
	glutDisplayFunc ( display_func );

	// create menu
    glutCreateMenu(main_menu_func);
    glutAddMenuEntry("Calculate Closest Distance using CPU", 0);
    glutAddMenuEntry("Calculate Closest Distance using CUDA", 1);
	glutAddMenuEntry("Update Meshes", 2);
	glutAddMenuEntry("Run 100 cycles cuda test", 3);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

bool import_stl(char f_name[], float3* &tri_list, int &tri_count)
{
	FILE * iFile1 = NULL;
	iFile1 = fopen(f_name, "r");

	char vname[] = "vertex";
	char line[256];
	int num = 0;

	while(get_line(iFile1, line, 256) != EOF)
	{
		num++;
	}

	tri_count = (int)((num) / 7);
	fseek(iFile1, 0, SEEK_SET);
	num = 0;
	tri_list = (float3 *) malloc(sizeof(float3) * tri_count * 3);
	int correct_count = 0;
	get_line(iFile1, line, 256);
	while(num < tri_count)
	{
		get_line(iFile1, line, 256);
		get_line(iFile1, line, 256);
		correct_count += fscanf(iFile1, "%s %f %f %f", vname, &tri_list[num].x, &tri_list[num].y, &tri_list[num].z);
		correct_count += fscanf(iFile1, "%s %f %f %f", vname, &tri_list[num + tri_count].x, &tri_list[num + tri_count].y, &tri_list[num + tri_count].z);
		correct_count += fscanf(iFile1, "%s %f %f %f", vname, &tri_list[num + tri_count * 2].x, &tri_list[num + tri_count * 2].y, &tri_list[num + tri_count * 2].z);
		num++;
		get_line(iFile1, line, 256);
		get_line(iFile1, line, 256);
		get_line(iFile1, line, 256);
	}
	fclose(iFile1);
	if(correct_count == tri_count * 12)
		return true;
	else
		return false;

}



int main(int argc, char **argv)
{
	win_x = 800;
	win_y = 800;
	
	bool check1;
	bool check2;

	check1 = import_stl("object1.stl", h_tri_list1, tri_count1);
	check2 = import_stl("object2.stl", h_tri_list2, tri_count2);

	if(check1)
		printf("First object imported.\n");
	else
	{
		printf("A problem occured while loading the first file please check. Please check file named \"object1.stl\" \n");
		return 0;
	}
	if(check1)
		printf("Second object imported.\n");
	else
	{
		printf("A problem occured while loading the second file please check. Please check file named \"object2.stl\" \n");
		return 0;
	}

	
	cam_pos.z = -20;
	glutInit(&argc, argv);
	open_glut_window();
	glutMainLoop();

	return 0;
}
