#include <math.h> //sqrt
#include <omp.h> //omp_get_wtime
#include <stdio.h> //fopen, fread, fclose
#include <string.h> //memset

#include <chrono> //std::chrono::steady_clock
#include <fstream> //ifstream
#include <iomanip> //setprecision
#include <iostream> //cout
#include <sstream> //stringstream
#include <vector> //vector
#include <algorithm>

using namespace std; 

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0; //v_num表示图的节点数
int e_num = 0; //e_num表示图的边数 
int F0 = 0, F1 = 0, F2 = 0; //F0表示输入层的维度，F1表示隐藏层的维度，F2表示输出层的维度

vector<vector<int>> edge_index; //edge_index[i][j]表示节点i的第j个邻居节点
vector<vector<float>> edge_val; //edge_val[i][j]表示节点i到节点edge_index[i][j]的边的权重
vector<int> degree; //degree[i]表示节点i的度
vector<int> raw_graph; //raw_graph[i]表示边的起点和终点

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter; //分别表示输入层的特征矩阵，第一层的权重矩阵，第二层的权重矩阵，第一层的特征矩阵，第一层的特征矩阵的中间结果，第二层的特征矩阵，第二层的特征矩阵的中间结果



//constructs an adjacency list in the CSR format from a raw graph
void construct_adjacency_list_csr(const std::vector<int>& raw_graph, std::vector<int>& row_ptr, std::vector<int>& col_idx) {
  int num_edges = raw_graph.size() / 2;
  int num_vertices = *std::max_element(raw_graph.begin(), raw_graph.end()) + 1;

  // Compute the degree of each vertex
  std::vector<int> degree(num_vertices, 0);
  for (int i = 0; i < num_edges; i++) {
    int src = raw_graph[2 * i];
    degree[src]++;
  }

  // Compute the row pointer array
  row_ptr.resize(num_vertices + 1);
  row_ptr[0] = 0;
  for (int i = 0; i < num_vertices; i++) {
    row_ptr[i + 1] = row_ptr[i] + degree[i];
  }

  // Compute the column index array
  col_idx.resize(num_edges);
  std::vector<int> next(num_vertices, 0);
  for (int i = 0; i < num_edges; i++) {
    int src = raw_graph[2 * i];
    int dst = raw_graph[2 * i + 1];
    col_idx[row_ptr[src] + next[src]] = dst;
    next[src]++;
  }
}

//readGraph(char* fname)：从文件中读取图的节点和边的信息，存储为邻接表形式(不进行修改)
void readGraph(char *fname) { //读取图的节点和边的信息，存储为邻接表形式
  ifstream infile(fname); //打开文件

  int source; //source表示边的起点
  int end; //end表示边的终点

  infile >> v_num >> e_num; //读取节点数和边数

  // raw_graph.resize(e_num * 2);

  while (!infile.eof()) { //读取边的信息
    infile >> source >> end; //读取边的起点和终点
    if (infile.peek() == EOF) break; //如果读到文件末尾，就退出循环
    raw_graph.push_back(source); //将边的起点存储到raw_graph中
    raw_graph.push_back(end); //将边的终点存储到raw_graph中
  }
}

//将边信息转化为邻接表形式的图结构，也就是这里面的edge_index和degree
void raw_graph_to_AdjacencyList() {
  int src; //src表示边的起点
  int dst; //dst表示边的终点

  edge_index.resize(v_num);// edge_index[i][j]表示节点i的第j个邻居节点
  degree.resize(v_num, 0); // degree[i]表示节点i的度

#pragma omp parallel for private(src, dst) shared(raw_graph, edge_index, degree)
  for (int i = 0; i < raw_graph.size() / 2; i++) { //遍历所有的边
    src = raw_graph[2 * i]; //src表示边的起点
    dst = raw_graph[2 * i + 1]; //dst表示边的终点
#pragma omp critical
    {
      edge_index[dst].push_back(src); //将边的起点存储到edge_index中
      degree[src]++; //将边的起点的度加1
    }
  } 
}

void edgeNormalization() { //对边进行归一化
  edge_val.resize(v_num); // edge_val[i][j]表示节点i到节点edge_index[i][j]的边的权重

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) { //遍历所有的节点
    for (int j = 0; j < edge_index[i].size(); j++) { //遍历节点i的所有邻居节点
      float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]); //计算边的权重
#pragma omp critical
      {
        edge_val[i].push_back(val); //将边的权重存储到edge_val中
      }
    }
  }
}
//从文件中读取float类型的数据(不进行修改)
void readFloat(char *fname, float *&dst, int num) { 
  dst = (float *)malloc(num * sizeof(float)); //为dst分配内存
  FILE *fp = fopen(fname, "rb"); //打开文件
  fread(dst, num * sizeof(float), 1, fp); //读取数据
  fclose(fp); //关闭文件
}

void initFloat(float *&dst, int num) { //初始化float类型的数据
  dst = (float *)malloc(num * sizeof(float)); //为dst分配内存
  memset(dst, 0, num * sizeof(float)); //将dst中的数据全部置为0 
  //为什么要全部置为0？因为这里的dst是一个矩阵，矩阵中的元素都是0，所以这里要将dst中的数据全部置为0
} 

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W) { //计算XW
  float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X; //将in_X转化为二维数组
  float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X; //将out_X转化为二维数组
  float(*tmp_W)[out_dim] = (float(*)[out_dim])W; //将W转化为二维数组

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) { //遍历所有的节点
    for (int j = 0; j < out_dim; j++) { //遍历节点i的所有邻居节点
      for (int k = 0; k < in_dim; k++) { //遍历节点i的所有邻居节点
        tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j]; //计算XW
      }
    }
  }
}

void AX(int dim, float *in_X, float *out_X) { //计算AX
  float(*tmp_in_X)[dim] = (float(*)[dim])in_X; //将in_X转化为二维数组
  float(*tmp_out_X)[dim] = (float(*)[dim])out_X; //将out_X转化为二维数组

#pragma omp parallel for
  for (int i = 0; i < v_num; i++) { //遍历所有的节点
    vector<int> &nlist = edge_index[i]; //获取节点i的所有邻居节点
    for (int j = 0; j < nlist.size(); j++) { //遍历节点i的所有邻居节点
      int nbr = nlist[j]; //获取节点i的第j个邻居节点
      for (int k = 0; k < dim; k++) { //遍历节点i的所有邻居节点
        tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j]; //计算AX
      }
    }
  }
}

void ReLU(int dim, float *X) { //计算ReLU
#pragma omp parallel for
  for (int i = 0; i < v_num * dim; i++) //遍历所有的节点
    if (X[i] < 0) X[i] = 0; //计算ReLU
}

void LogSoftmax(int dim, float *X) { //计算LogSoftmax
  float(*tmp_X)[dim] = (float(*)[dim])X; //将X转化为二维数组
#pragma omp parallel for
  for (int i = 0; i < v_num; i++) { //遍历所有的节点
    float max = tmp_X[i][0]; //获取节点i的第一个元素
    for (int j = 1; j < dim; j++) { //遍历节点i的所有元素
      if (tmp_X[i][j] > max) max = tmp_X[i][j]; //获取节点i的最大元素
    }

    float sum = 0; //初始化sum
    for (int j = 0; j < dim; j++) { //遍历节点i的所有元素
      sum += exp(tmp_X[i][j] - max); //计算LogSoftmax
    }
    sum = log(sum); //计算LogSoftmax

    for (int j = 0; j < dim; j++) { //遍历节点i的所有元素
      tmp_X[i][j] = tmp_X[i][j] - max - sum; //计算LogSoftmax
    }
  }
}

float MaxRowSum(float *X, int dim) { //计算MaxRowSum
  float(*tmp_X)[dim] = (float(*)[dim])X; //将X转化为二维数组
  float max = -__FLT_MAX__; //初始化max
#pragma omp parallel for reduction(max:max)
  for (int i = 0; i < v_num; i++) { //遍历所有的节点
    float sum = 0; //初始化sum
    for (int j = 0; j < dim; j++) { //遍历节点i的所有元素
      sum += tmp_X[i][j]; //计算MaxRowSum
    } 
    if (sum > max) max = sum; //计算MaxRowSum
  }
  return max; //返回max
}

void freeFloats() {
  free(X0); 
  free(W1);
  free(W2);
  free(X1);
  free(X2);
  free(X1_inter);
  free(X2_inter);
}

void somePreprocessing() {
  // The graph  will be transformed into adjacency list, you can use other data
  // structure such as CSR
  raw_graph_to_AdjacencyList(); //将原始图转化为邻接表
}

int main(int argc, char **argv) {
  // Do NOT count the time of reading files, malloc, and memset
  F0 = atoi(argv[1]); //获取F0
  F1 = atoi(argv[2]); //获取F1
  F2 = atoi(argv[3]); //获取F2

  readGraph(argv[4]); //读取图
  readFloat(argv[5], X0, v_num * F0); //读取X0
  readFloat(argv[6], W1, F0 * F1); //读取W1
  readFloat(argv[7], W2, F1 * F2); //读取W2

  initFloat(X1, v_num * F1); //初始化X1
  initFloat(X1_inter, v_num * F1); //初始化X1_inter
  initFloat(X2, v_num * F2); //初始化X2
  initFloat(X2_inter, v_num * F2); //初始化X2_inter

  // Time point at the start of the computation
  TimePoint start = chrono::steady_clock::now();  
 
  // Preprocessing time should be included

  somePreprocessing(); //预处理

  edgeNormalization(); //边归一化

  // printf("Layer1 XW\n");
  XW(F0, F1, X0, X1_inter, W1); //计算XW

  // printf("Layer1 AX\n");
  AX(F1, X1_inter, X1); //计算AX

  // printf("Layer1 ReLU\n");
  ReLU(F1, X1); //计算ReLU

  // printf("Layer2 XW\n");
  XW(F1, F2, X1, X2_inter, W2); //计算XW

  // printf("Layer2 AX\n");
  AX(F2, X2_inter, X2); //计算AX

  // printf("Layer2 LogSoftmax\n");
  LogSoftmax(F2, X2); //计算LogSoftmax 

  // You need to compute the max row sum for result verification
  float max_sum = MaxRowSum(X2, F2); //计算MaxRowSum

  // Time point at the end of the computation
  TimePoint end = chrono::steady_clock::now(); 
  chrono::duration<double> l_durationSec = end - start; 
  double l_timeMs = l_durationSec.count() * 1e3; //计算时间

  // Finally, the max row sum and the computing time
  // should be print to the terminal in the following format
  printf("%.8f\n", max_sum);  
  printf("%.8lf\n", l_timeMs);

  // Remember to free your allocated memory
  freeFloats();
}