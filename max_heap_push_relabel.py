
import os
import time
import random
import pandas as pd
from colorama import Fore
from collections import defaultdict
import time
from collections import deque
import heapq
NUM_OF_TEST_CASES = 1
INF = float('Inf')


class Vertex:
    def __init__(self, i, w):
        # number of the end vertex
        # weight or capacity
        # associated with the edge
        self.i = i
        self.w = w


class DirectedGraph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adjacencyList = [[] for i in range(vertices)]

    def addEdge(self, u, v, weight):
        self.adjacencyList[u].append(Vertex(v, weight))

    def hasEdge(self, u, v):
        if u >= self.vertices:
            return False

        for vertex in self.adjacencyList[u]:
            if vertex.i == v:
                return True
        return False

    def getEdge(self, u, v) -> Vertex | None:
        for vertex in self.adjacencyList[u]:
            if vertex.i == v:
                return vertex
        return None


class MaxFlow:
    def __init__(self, graph: DirectedGraph, source, sink):
        self.graph = graph
        self.source = source
        self.sink = sink
        self.n = self.graph.vertices
        self.heap = []

    def initResidualGraph(self):
        self.residualGraph = DirectedGraph(self.graph.vertices)
        for u in range(self.graph.vertices):
            for v in self.graph.adjacencyList[u]:
                if self.residualGraph.hasEdge(u, v.i):
                    self.residualGraph.getEdge(u, v.i).w += v.w
                else:
                    self.residualGraph.addEdge(u, v.i, v.w)
                if not self.residualGraph.hasEdge(v.i, u):
                    self.residualGraph.addEdge(v.i, u, 0)

    def HighestLabelPushRelabel(self):
        self.initResidualGraph()
        self.excess = [0]*self.graph.vertices
        self.height = [0]*self.graph.vertices
        self.excess[self.source] = INF
        self.height[self.source] = self.n
        for v in self.graph.adjacencyList[self.source]:
            self.excess[v.i] += v.w
            self.residualGraph.getEdge(self.source, v.i).w = 0
            self.residualGraph.getEdge(v.i, self.source).w = v.w

        self.build_heap()
        while self.heap:

            current = heapq.heappop(self.heap)
            self.push(current[1])
            if self.excess[current[1]] > 0:
                self.relabel(current[1])

        return self.excess[self.sink]

    def build_heap(self):
        for i in range(self.n):
            if i != self.source and i != self.sink and self.excess[i] > 0:
                # print(f'pusing to heap {i}')
                heapq.heappush(self.heap, [-self.height[i], i])

    def relabel(self, u):
        minHeight = float("inf")
        relabeled = False
        for v in self.residualGraph.adjacencyList[u]:
            if v.w > 0:

                minHeight = min(minHeight, self.height[v.i])
                self.height[u] = minHeight + 1
                relabeled = True
        if relabeled:
            heapq.heappush(self.heap, [-self.height[u], u])

    def push(self, u):
        for v in self.residualGraph.adjacencyList[u]:
            # after pushing flow if
            # there is no excess flow,
            # then break
            if self.excess[u] == 0:
                break

            # push more flow to
            # the adjacent v if possible
            if v.w > 0 and self.height[v.i] < self.height[u]:
                # flow possible
                f = min(self.excess[u], v.w)

                v.w -= f
                self.residualGraph.getEdge(v.i, u).w += f

                self.excess[u] -= f
                self.excess[v.i] += f

                if not v.i in self.heap and v.i != self.source and v.i != self.sink:
                    heapq.heappush(self.heap, [-self.height[v.i], v.i])


class Graph:
    def __init__(self, graph, verbose=False):
        n = len(graph[0])
        self.dg = DirectedGraph(n)
        for row in range(n):
            for col in range(n):
                if graph[row][col] != 0:
                    self.dg.addEdge(row, col, graph[row][col])

    def push_relabel(self, s, t):
        maxflow = MaxFlow(self.dg, s, t)
        return maxflow.HighestLabelPushRelabel()


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
                if i == j or i == self.t or j == self.s:
                    row.append(0)
                else:
                    row.append(random.choices(
                        range(int(self.n/2)+1), weights=prob, k=1)[0])
            matrix.append(row)

        df = pd.DataFrame(matrix)
        df.to_csv('dataset/graph-{}.csv'.format(id))

    def test(self, id):
        # Read data from file
        df = pd.read_csv('dataset/graph-{}.csv'.format(id), index_col=0)
        graph = df.values

        start = time.time()
        g = Graph(graph)
        result = g.push_relabel(self.s, self.t)
        end = time.time()

        print('Test {} using Push-relabel algorithm with Max-heap selection runs within {} ms'.format(id,
              (end-start) * 10**3))

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
    # capacity = [[0, 5, 3, 0], [0, 0, 0, 7], [0, 2, 0, 0], [0, 0, 0, 0]]

    # graph = Graph(capacity)
    # print(graph.push_relabel(0, 3))
    # return
    for i in range(10):
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
