# Boruvka algorithm parallel implementation using CUDA

Works for matrix represantion of connected, non-directed graph, where all edges are integers and non-existing edges are equal to INT_MAX

Input:
integer n - number of vertices
matrix n*n G, where G_ij is the distance of edge between i and j
(Proper input may be created by graphGenerator)

Output:
integer result - sum of edges used in minimum spanning tree of the graph


The same input and output works for Prim Algorithm implementation. Used to testing.
