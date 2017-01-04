#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <climits>

#define MAX_SIZE 10000             //max nr of vertices
#define ST first
#define ND second

using namespace std;

int Prim();
void readData();
int primMST();

int n;
int D[MAX_SIZE][MAX_SIZE];

int main(){
        readData();
        cout << Prim() << endl;
}


void readData(){
    cin >> n;
    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            cin >> D[i][j];
        }
    }
}


int Prim(){
    int MSTleft = n;
    int result = 0;
    vector<bool> MSTvisited;
    MSTvisited.resize(n, false);
    //PRIM MST ALGORITHM
    priority_queue<pair<int, int>, vector<pair<int, int>>> PQ;

    for(int i=0; i<n; i++){                 //chosing first vertex
        if(MSTvisited[i]) continue;

        MSTvisited[i] = true;
        MSTleft--;
        for(int j=0; j<n; j++){
            if(MSTvisited[j]) continue;
            PQ.emplace(make_pair(-D[i][j], j));
        }
        break;
    }

    while(MSTleft){
        pair<int, int> P = PQ.top();
        PQ.pop();
        if (MSTvisited[P.ND]) continue;

        result -= P.ST;
        MSTvisited[P.ND] = true;
        MSTleft--;

        for(int i=0; i<n; i++){
            if(MSTvisited[i]) continue;
            PQ.emplace(make_pair(-D[P.ND][i], i));
        }
    }
    return result;
}
