#include "boruvka.h"
#include <cstdio>
#include <cstdlib>
using namespace std;

int main(){
    int n;
    scanf("%d", &n);

    int * G = (int*) malloc(n*n*sizeof(int));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            scanf("%d", &G[i*n+j]);
        }
    }
 
    int result = boruvka(n, G);

    printf("%d\n", result);

    free(G);
    return 0;
}