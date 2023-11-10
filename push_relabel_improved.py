
import os
import time
import random
import pandas as pd
from colorama import Fore
from collections import defaultdict
import time
NUM_OF_TEST_CASES = 1
INF = float('Inf')
class Graph:
    def __init__(self, graph):
        self.residual = graph
        self.n = len(graph)
        self.height = [0]*self.n
        self.excess = [0]*self.n

    def push(self, u, v):

        d = min(self.excess[u], self.residual[u][v])
        self.residual[u][v] -= d
        self.residual[v][u] += d
        self.excess[u] -= d
        self.excess[v] += d
        # print(f'push {u} to {v}')
        # print(f'    flow: {self.flow}')
        # print(f'    excess: {self.excess}')
        # time.sleep(3)
    
    def relabel(self, u):
        d = INF
        for i in range(self.n):
            if self.residual[u][i]> 0:
                d = min (d, self.height[i])
        
        if d < INF:
            self.height[u] = d + 1
        # print (f'relabel {u}')
        # print(f'    {self.height}')
        # time.sleep(3)
    
    def find_max_height_vertices(self, s, t):
        max_height = []
        for i in range(self.n):
            if i != s and i != t and self.excess[i] >  0:
                if (max_height and self.height[i] > self.height[max_height[0]]):
                    max_height.clear()
                if (not max_height or self.height[i] == self.height[max_height[0]]):
                    max_height.append(i)
        
        # print(f'height {self.height} find_max_height_vertices: {max_height}, {s}, {t}')
        # time.sleep(3)
        return max_height


    def push_relabel(self, s, t):
        self.height[s] = self.n
        self.flow = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.excess = [0]*self.n
        self.excess[s] = INF

        for i in range(self.n):
            if (i != s):
                self.push(s, i)

        current = self.find_max_height_vertices(s,t)
        while current:
            # print(f'current max_height_verices: {current}')
            for i in current:
                # print(f'    working on {i}')
                pushed = False
                for j in range(self.n):
                    if self.excess[i] <= 0:
                        # print(f'        excess of {i} lower than 0')
                        break
                    if (self.residual[i][j] > 0 and self.height[i] == self.height[j] + 1):
                        # print(f'        pusing from {i} to {j}' ,self.capacity[i][j], self.flow[i][j], self.height[i], self.height[j])
                        self.push(i,j)
                        pushed = True

                if not pushed:
                    self.relabel(i)
                    break

            current = self.find_max_height_vertices(s,t)

        # print(self.residual)
        return sum(self.residual[t])






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
        result = g.push_relabel(self.s, self.t)
        end = time.time()
        
        print('Test {} using Push-relabel algorithm runs within {} ms'.format(id, (end-start) * 10**3))
        
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
    # capacity = [[0,5,3,0],[0,0,0,7],[0,2,0,0],[0,0,0,0]]

    # graph = Graph(capacity)
    # print(graph.push_relabel(0,3))
    # return
    for i in range(6,10):
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