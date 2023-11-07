import os
import time
import random
import pandas as pd
from colorama import Fore
from collections import defaultdict

NUM_OF_TEST_CASES = 10

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    # Using BFS as a searching algorithm 
    def searching_algo_BFS(self, s, t, parent):

        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    # Applying fordfulkerson algorithm
    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.searching_algo_BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Adding the path flows
            max_flow += path_flow

            # Updating the residual values of edges
            v = sink
            while(v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


class Testcase:
    def __init__(self, num_of_vertices, source, sink):
        self.n = num_of_vertices
        self.s = source
        self.t = sink

    def generate_data_test(self, id):
        prob = [1 for i in range(int(self.n/2)+1)]
        prob[0] = self.n/2

        matrix = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i==j or i==self.t or j==self.s:
                    row.append(0)
                else:
                    row.append(random.choices(range(int(self.n/2)+1), weights=prob, k=1)[0])
            matrix.append(row)

        df = pd.DataFrame(matrix)
        df.to_csv('dataset/graph-{}.csv'.format(id))

    def test(self, id):
        # Read data from file
        df = pd.read_csv('dataset/graph-{}.csv'.format(id), index_col=0)
        graph= df.values
        
        start = time.time()
        g = Graph(graph)
        result = g.ford_fulkerson(self.s, self.t)
        end = time.time()
        
        print('Test {} using Ford-Fulkerson algorithm runs within {} ms'.format(id, (end-start) * 10**3))
        
        f = open('testcases/actual_output/{}.txt'.format(id), 'w')
        f.write(str(result))
        f.close()
        
        f = open('testcases/expected_output/{}.txt'.format(id), 'r')
        if result == int(f.read()):
            print('\033[92m{}\033[00m'.format('Passed'))
        else:
            print('\033[91m{}\033[00m'.format('Failed'))
        f.close()
        

def main():
    for i in range(NUM_OF_TEST_CASES):
        path = 'testcases/input/{}.txt'.format(i+1)
        f = open(path, 'r')
        inputs = f.read().splitlines()
        N, source, sink = int(inputs[0]), int(inputs[1]), int(inputs[2])
        
        tc = Testcase(N, source, sink)
        if not os.path.isfile('dataset/graph-{}.csv'.format(i+1)):
            print("Hello")
            tc.generate_data_test(i+1)
        tc.test(i+1)
        

if __name__ == "__main__":
    main()