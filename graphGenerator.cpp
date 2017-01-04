#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cstdlib>

using namespace std;

#define MP make_pair
#define ST first
#define ND second



int main(){
    ios_base::sync_with_stdio(false);
    int n, seed;
    cin >> n >> seed;
    srand(seed);
    cout << n << endl;
    int G[n][n];

    for(int i=0; i<n; i++){
        for(int j=i; j<n; j++){
            if (i==j) G[i][i] = INT_MAX;
            else G[j][i] = G[i][j] = rand()%999999+1    ;
        }
    }
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            cout << G[i][j] << " ";
        }
        cout << endl;
    }
}
