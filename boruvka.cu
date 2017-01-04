#include <cstdio>
#include <algorithm>
#include <climits>

extern "C" {

	int min(int a, int b){
		return a < b ? a : b; 
	}

	__global__
    void set(int* parent, int* newParent, int *res, int n){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(thid >= n) return;

        parent[thid] = newParent[thid] = thid;
        res[thid] = 0;
    }


    //one thread - one vertex
    //find the closest neighbour for each vertex
	__global__
	void closest_neighbour(int* G, int* parent, int* newParent, int* roundNeighbourDist, int* toSubtract, int n){
		int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		
        if (thid >= n) return;
        if (parent[thid] != thid) return;

        int minimum = INT_MAX;
        int neighbour = -1;

		for(int i=0; i<n; i++){
            int dist = G[thid+i*n];
            if(dist >= minimum) continue;

            minimum = dist;
            neighbour = parent[i];
        }

        if (neighbour == -1) return;

        toSubtract[thid] = roundNeighbourDist[thid] = minimum;


        //start vertex union
        while(true){
            while(thid != newParent[thid]) thid = newParent[thid];
            while(neighbour != newParent[neighbour]) neighbour = newParent[neighbour];
            if(thid < neighbour){
                if (neighbour == atomicCAS(&newParent[neighbour], neighbour, thid)) {
                    break;
                }
            }
            else{
                if (thid == atomicCAS(&newParent[thid], thid, neighbour)){
                    break;
                } 
            }
        }
    }

    //one thread - one vertex
    //find component root for every vertex
    //sum MST for component in root
    __global__
    void unionn(int* parent, int* newParent, int* MSTres, int* roundNeighbourDist, int* toSubtract, int n, int* goOn){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        if (thid >= n) return;

        int parentToBe = newParent[parent[thid]];

        while(parentToBe != newParent[parentToBe]) parentToBe = newParent[parentToBe];

        if(thid == parent[thid] && parentToBe != thid){ //previously root, now part of new component
            atomicAdd(&MSTres[parentToBe], MSTres[thid]);
            atomicAdd(&MSTres[parentToBe], roundNeighbourDist[thid]);
            atomicMin(&toSubtract[parentToBe], roundNeighbourDist[thid]);
        }
        else if(parentToBe == thid){ //new component root
            atomicAdd(&MSTres[parentToBe], roundNeighbourDist[thid]);
            atomicMin(&toSubtract[parentToBe], roundNeighbourDist[thid]);
        }

        newParent[thid] = parent[thid] = parentToBe;

        //finally all vertices should belong to component 0        
        if(parent[thid] != 0){
            *goOn = 1;
        }
    }

    //subtract the repeated root in each new component
    __global__
    void subtract(int *parent, int * MSTres, int* toSubtract, int n){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        if (thid >= n) return;
        if (parent[thid] != thid) return;

        MSTres[thid] -= toSubtract[thid];
    }

    __global__
    void transpose(int* A, int* AT, int N){
        int M = N;
        int thidx = (blockIdx.x * blockDim.x) + threadIdx.x;
        int thidy = (blockIdx.y * blockDim.y) + threadIdx.y;
        
        __shared__ int T[32][33];
        
        if(thidx < M & thidy < N)
            T[threadIdx.y][threadIdx.x] = A[thidy*M+thidx];
        __syncthreads();


        thidx = (blockIdx.y * blockDim.x) + threadIdx.x;
        thidy = (blockIdx.x * blockDim.y) + threadIdx.y;
        
        if(thidx < N & thidy < M)
            AT[thidy*N+thidx] = T[threadIdx.x][threadIdx.y];
    }

    //one thread - one vertex (destination)
    //choose the smallest distance from each component to the vertex
    __global__
    void merge(int *G, int *parent, int n){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        
        if (thid >= n) return;

        int thidGroup = parent[thid];

        for(int i=0; i<n; i++){
            if(i == thid){ //set distance to self as INT_MAX
                G[thidGroup*n+thid] = INT_MAX;
            }
            else{
                int iGroup = parent[i];
                if(iGroup != thidGroup){
                    G[iGroup*n+thid] = min(G[iGroup*n+thid], G[i*n+thid]);
                }                
            }
        }
    }
}