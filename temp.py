import numpy as np
import networkx as nx

A = np.loadtxt("karate.txt", delimiter=",")

G = nx.from_numpy_array(A, create_using=nx.DiGraph)
pr = nx.pagerank(G, alpha=1, tol=1e-10)  # power-iteration under the hood

top10 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 PageRank (1-based node ids):")
for node, score in top10:
    print(node+1, score)