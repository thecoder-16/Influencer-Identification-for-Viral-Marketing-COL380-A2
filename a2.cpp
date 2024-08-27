#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

void print_graph(vector<vector<int>>& graph){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
        for(int i=0; i<graph.size(); i++){
            cout<<i<<" : ";
            for(auto &e: graph[i]) cout<<e<<" "; 
            cout<<endl;
        }
}

void bfs(int vertex,vector<int>& visited,vector<vector<int>> graph,vector<int>& comp){
    int n = graph.size();
    queue<int> q;
    q.push(vertex);
    visited[vertex] = 1;
    while(!q.empty()){
        int curr = q.front();
        comp.push_back(curr);
        q.pop();
        for(int i : graph[curr]){
            if(!visited[i]){
                q.push(i);
                visited[i] = 1;
            }
        }
    }

}

void conComp(vector<vector<int>> graph, vector<vector<int>>& comps){
    int n = graph.size();
    vector<int> visited(n,0);
    
    for(int i = 0;i < n;i++){
            if(graph[i].size() == 0){
                visited[i] = 1;
        } 
    }
    for (int v = 0; v < n; v++) {
        if (visited[v] == 0) {
            vector<int> comp;
            bfs(v,visited,graph,comp);
            if(comp.size()>1) {
                comps.push_back(comp);
            }
        }
    }
}

int intersection(int u, int v, vector<vector<int>>& graph) {
    int sup = 0;
    for(int w : graph[u]){
        if(w != v && find(graph[v].begin(),graph[v].end(),w) != graph[v].end()){
            sup++;
        }
    }
    return sup;
}

void prefilter(vector<vector<int>>& graph, int k){
    queue<int> deletable;
    for(int i=0; i<graph.size(); i++)
        if(graph[i].size()<k-1)
            deletable.push(i);

    while(!deletable.empty()){
        int v = deletable.front();
        deletable.pop();
        for(int &u : graph[v]){
            auto it_v = std::lower_bound(graph[u].begin(), graph[u].end(), v);
            if(it_v != graph[u].end()){
                graph[u].erase(it_v);
            }
            if(graph[u].size()<k-1){
                deletable.push(u);
            }
        }
        graph[v].clear();
    }
}

void edge_delete(vector<vector<int>>& graph, int x, int y){
    auto it_x = std::lower_bound(graph[x].begin(), graph[x].end(), y);
    auto it_y = std::lower_bound(graph[y].begin(), graph[y].end(), x);
    if(it_x != graph[x].end() && *it_x == y)
        graph[x].erase(it_x);
    if(it_y != graph[y].end() && *it_y == x)
        graph[y].erase(it_y);
}

bool FilterEdges(vector<vector<int>>& graph, int k){
    bool flag = false;
    int n = graph.size();
    vector<pair<int, int>> deletable;

    int rank, size, rank_done = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int chunk_size = (n + size - 1) / size;

    for(int i=rank; i<n; i+=size)
        for (int j : graph[i]) 
            if (i < j) {
                int common_neighbors = intersection(i, j, graph);
                if(common_neighbors < k-2)
                    deletable.push_back({i, j});
            }

    int max_size;
    int qq = deletable.size()*2;
    MPI_Allreduce(&qq, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    vector<int> del_ne;
    for(auto &e:deletable){
        del_ne.push_back(e.first);
        del_ne.push_back(e.second);
    }
    while(del_ne.size() != max_size){
        del_ne.emplace_back(-1);
    }
        
    for(int i = 0; i<(max_size+49)/50; i++){
        int all_edges[50*size];
        memset(all_edges, -1, sizeof(all_edges));
        int ssize = min(50, max_size-i*50);
        MPI_Allgather(&del_ne[i*50], ssize, MPI_INT, &all_edges, ssize, MPI_INT, MPI_COMM_WORLD);

        for(int j=0; j<ssize*size; j+=2){  
            if(all_edges[j]==-1){
                continue;
            }
            if(all_edges[j]%size != rank){
                edge_delete(graph, all_edges[j], all_edges[j+1]);
            }
            
        }
    }

    while(max_size>0){
        flag = true;
        vector<pair<int, int>> del_new;
        for(int w=0; w<deletable.size(); w++){
            pair<int, int> e = deletable[w];
            if(e.first<0 or e.second<0) continue;
            edge_delete(graph, e.first, e.second);
                
            for (int v : graph[e.first]) {
                if (v != e.second && find(graph[e.second].begin(), graph[e.second].end(), v) != graph[e.second].end()) {
                    pair<int, int> p = {min(v, e.first),  max(v, e.first)};
                    int supp = 0; 
                    for (int w : graph[p.first])
                        if(w != p.second && find(graph[p.second].begin(), graph[p.second].end(), w) != graph[p.second].end())
                            supp++;
                    if(supp < k-2 && find(del_new.begin(), del_new.end(), p) == del_new.end() && find(deletable.begin(), deletable.end(), p) == deletable.end() )
                        del_new.emplace_back(p);
                    p = {min(v, e.second), max(v, e.second)};
                    supp = 0; 
                    for (int w : graph[p.first])
                        if(w != p.second && find(graph[p.second].begin(), graph[p.second].end(), w) != graph[p.second].end())
                            supp++;
                    if(supp < k-2 && find(del_new.begin(), del_new.end(), p) == del_new.end() && find(deletable.begin(), deletable.end(), p) == deletable.end())
                        del_new.emplace_back(p);
                }
            }
        }

        deletable.clear();

        qq = del_new.size()*2;
        MPI_Allreduce(&qq, &max_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // update del_new to maxsize
        vector<int> del_ne1;
        for(auto &e:del_new){
            del_ne1.push_back(e.first);
            del_ne1.push_back(e.second);
        }
        while(del_ne1.size() != max_size){
            del_ne1.emplace_back(-1);
        }
        
        int all_edges[50*size];
        memset(all_edges, -1, sizeof(all_edges));
        for(int i = 0; i<(max_size+49)/50; i++){
            int ssize = min(50, max_size-i*50);
            MPI_Allgather(&del_ne1[i*50], ssize, MPI_INT, &all_edges, ssize, MPI_INT, MPI_COMM_WORLD);

            for(int j=0; j<50*size; j+=2){  
                if(all_edges[j]==-1)
                    continue;                    
                if(all_edges[j]%size == rank){
                    deletable.push_back({all_edges[j], all_edges[j+1]});
                }
                else    
                   edge_delete(graph, all_edges[j], all_edges[j+1]);
            }
        }
    }
    return flag;
}


void k_truss(vector<vector<int>>& graph, int k){
    int n = graph.size();
    auto start = high_resolution_clock::now();

    prefilter(graph, k);
    bool flag = true;
    while(flag){
        flag = FilterEdges(graph, k); 
    }

    prefilter(graph, k);
}

void read_input_matrix(string filename, vector<vector<int>>& graph) {
    ifstream input_file(filename, ios::binary);
    int n, m;
    input_file.read((char*)&n, 4);
    input_file.read((char*)&m, 4);
    graph.resize(n);
    for (int i = 0; i < n; i++) {
        int node_num = 0,deg = 0;
        input_file.read((char*)&node_num, 4);
        input_file.read((char*)&deg, 4);
        graph[node_num].resize(deg);
        
        for (int l = 0; l < deg; l++) {
            int nbour;
            input_file.read((char*)&nbour, 4);
            graph[node_num][l] = nbour;
        }
    }
}

void output_concomp(string filename, vector<vector<int>>& k_truss_vertices, int verbose) {
    fstream of;
    of.open(filename, ios::app);

    if(k_truss_vertices.size()==0){
        of<<"0\n";
        of.close();
        return;
    }
    else 
        of<<"1\n";

    if(verbose==1) {
        int n = k_truss_vertices.size();
        of<<n<< endl;
        for (auto &v : k_truss_vertices) {
            sort(v.begin(), v.end());
            for(auto &e: v) of<<e<<" "; of<<endl;
        }
    }

    of.close();
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    int task_id, start_k, end_k, p, verbose;
    string input_file, header_file, output_file;
    
    for(int i=1; i < argc; i++){
        string s = argv[i];
        int j;
        for(j = 0; j < s.size();j++){
            if(s[j] == '='){
                break;
            }
        }
        string tag = s.substr(0,j);
        string val = s.substr(j+1,s.size());
        if (tag == "--taskid") {
            task_id = stoi(val);
        } 
        else if (tag == "--inputpath") {
            input_file = string(val);
        } 
        else if (tag == "--headerpath") {
            header_file = string(val);
        } 
        else if (tag == "--outputpath") {
            output_file = string(val);
        } 
        else if (tag == "--verbose") {
            verbose = stoi(val);
        } 
        else if (tag == "--startk") {
            start_k = stoi(val);
        } 
        else if (tag == "--endk") {
            end_k = stoi(val);
        } 
        else if (tag == "--p") {
            p = stoi(val);
        }
    }
    
    if(task_id == 1){
        vector<vector<int>> graph;
        read_input_matrix(input_file, graph);
        int n = graph.size();

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int k1 = start_k+2, k2 = end_k+2;
auto start = high_resolution_clock::now();
        for (int k = k1; k <= k2; k++) {
            k_truss(graph, k);
            vector<vector<int>> k_truss_vertices;
            conComp(graph, k_truss_vertices);
            if(rank==0)
                output_concomp(output_file, k_truss_vertices, verbose);
        }
auto end = high_resolution_clock::now();
auto duration = duration_cast<milliseconds>(end - start);
    
if(rank==0)
    cout << duration.count() << " ms" << endl;

        MPI_Finalize();
    }
    else{
        cout << "Invalid taskid\n";
    }


    return 0;
}

// mpic++ temp3.cpp -o m && mpirun -n 4 ./m --taskid=1 --inputpath="test-input-8.gra" --headerpath="hed.dat" --outputpath="ww.txt" --verbose=1 --startk=1 --endk=30 --p=1