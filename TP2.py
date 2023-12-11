import sys
import numpy as np
import networkx as nx
from queue import PriorityQueue
from timeout_decorator import timeout, TimeoutError
import time
import resource

'''
******************************************************************
leitura dos arquivos
******************************************************************
'''

def read_tsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    coordinates_section = False
    coordinates = []

    for line in lines:
        if line.startswith("NODE_COORD_SECTION"):
            coordinates_section = True
            continue
        elif line.startswith("EOF"):
            break

        if coordinates_section:
            parts = line.split()
            if len(parts) > 0:
                coordinates.append((float(parts[1]), float(parts[2])))
            else:
                break #caso especial para os arquivos sem EOF

    return coordinates

def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def create_distance_matrix(coordinates):
    num_vertices = len(coordinates)
    distance_matrix = np.zeros((num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in range(i, num_vertices):
            distance_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix

'''
******************************************************************
branch and bound
******************************************************************
'''
@timeout(1800)
def bnb_tsp(graph, optimum, best_path=None, best_value=sys.float_info.max):
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    n = len(graph)
    
    # Salavando o peso total das duas arestas de menor custo de cada vértice que serão usadas no 
    # lower bound para evitar que essa busca seja feita muitas vezes, economizando tempo.
    best_edges_weight = np.zeros(n)
    for i in range(n):
        best_1 = best_2 = sys.float_info.max
        for j in range(n):
            if i!=j:
                if graph[i, j] < best_2:
                    if graph[i, j] < best_1:
                        best_2 = best_1
                        best_1 = graph[i,j]
                    else:
                        best_2 = graph[i,j]
        best_edges_weight[i] = best_1 + best_2
    
    #Função de limite inferior (lower bound)
    def lower_bound(path, cur_cost):
        lb = 0
        for i in range(n):
            if i not in path:
                lb += best_edges_weight[i]
        lb /= 2 #cada aresta é contada duas vezes
        lb += cur_cost
        return lb
    
    # Nó na árvore de busca
    class Node:
        def __init__(self, path, bound, level, cost):
            self.path = path
            self.bound = bound
            self.level = level
            self.cost = cost

        def __lt__(self, other):
            return self.level > other.level or (self.level == other.level and self.bound < other.bound)
    
    # Inicialização
    initial_path = [0]
    initial_cost = 0
    initial_level = 0
    initial_bound = np.sum(best_edges_weight) / 2
    pq = PriorityQueue()
    pq.put(Node(initial_path, initial_bound, initial_level, initial_cost))
    
    # Início do Branch and Bound
    while not pq.empty():
        current_node = pq.get()
        
        if current_node.bound < best_value:
            if current_node.level < n-1:
                for i in range(1, n):
                    # Adiciona um novo nó para cada cidade não visitada
                    if i not in current_node.path:
                        new_path = current_node.path.copy()
                        new_path.append(i)
                        new_cost = current_node.cost + graph[new_path[-2], i]
                        new_bound = lower_bound(new_path, new_cost)
                        
                        # Adiciona o novo nó à fila de prioridade se ele tem a possibilidade de ser melhor que a melhor solução encontrada até agora
                        pq.put(Node(new_path, new_bound, current_node.level + 1, new_cost))
            else:
                new_path = current_node.path.copy()
                new_path.append(0)
                new_cost = current_node.cost + graph[new_path[-2], 0]
                if best_value > new_cost:
                    best_value = new_cost
                    best_path = new_path

    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_used = end_memory - start_memory
    end_time = time.time()
    elapsed_time = end_time - start_time
    return best_path, best_value, elapsed_time, memory_used, (best_value/optimum)

'''
******************************************************************
twice-around-the-tree
******************************************************************
'''
@timeout(1800)
def twice_around_the_tree(graph, optimum):
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    # Converte matriz de adjacências numpy em grafo do networkx
    G = nx.Graph(graph)
    mst = nx.minimum_spanning_tree(G)

    del G

    traversal = list(nx.dfs_preorder_nodes(mst, source=0))
    tour = [traversal[0]]
    for node in traversal[1:]:
        if node not in tour:
            tour.append(node)
    tour.append(traversal[0])  # Returning to the start

    del traversal
    
    cost = sum(graph[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_used = end_memory - start_memory
    end_time = time.time()
    elapsed_time = end_time - start_time
    return tour, cost, elapsed_time, memory_used, (cost/optimum)

'''
******************************************************************
Christofides
******************************************************************
'''
@timeout(1800)
def christofides_tsp(graph, optimum):
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    # Converte matriz de adjacências numpy em grafo do networkx
    G = nx.Graph(graph)
    
    # Passo 1: Obter uma árvore geradora mínima
    mst = nx.minimum_spanning_tree(G)

    # Passo 2: Encontrar os vértices de grau ímpar na árvore geradora mínima
    odd_degree_nodes = [node for node, degree in mst.degree() if degree % 2 == 1]

    # Passo 3: Encontrar um emparelhamento perfeito mínimo nos vértices de grau ímpar
    min_weight_matching = nx.Graph()
    
    for i in range(len(odd_degree_nodes)):
        for j in range(i + 1, len(odd_degree_nodes)):
            u, v = odd_degree_nodes[i], odd_degree_nodes[j]
            min_weight_matching.add_edge(u, v, weight=graph[u][v])
    
    matching = nx.min_weight_matching(min_weight_matching, maxcardinality=True)
    
    # Passo 4: Adicionar as arestas do emparelhamento à árvore geradora mínima
    augmented_mst = mst.copy()
    for edge in matching:
        augmented_mst.add_edge(*edge)

    # Passo 5: Encontrar um circuito euleriano na árvore aumentada (DFS)
    traversal = list(nx.dfs_preorder_nodes(augmented_mst, source=0))


    # Passo 6: Remover vértices duplicados e criar o ciclo hamiltoniano
    tour = [traversal[0]]
    for node in traversal[1:]:
        if node not in tour:
            tour.append(node)

    # Adicionar o primeiro vértice novamente para fechar o ciclo
    tour.append(traversal[0])  
    cost = sum(graph[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_used = end_memory - start_memory
    end_time = time.time()
    elapsed_time = end_time - start_time
    return tour, cost, elapsed_time, memory_used, (cost/optimum)


datasets = [
("eil51", 426), 
("berlin52", 7542), 
("st70", 675), 
("eil76", 538), 
("pr76", 108159), 
("rat99", 1211), 
("kroA100", 21282), 
("kroB100", 22141), 
("kroC100", 20749), 
("kroD100", 21294), 
("kroE100", 22068), 
("rd100", 7910), 
("eil101", 629), 
("lin105", 14379), 
("pr107", 44303), 
("pr124", 59030), 
("bier127", 118282), 
("ch130", 6110), 
("pr136", 96772), 
("pr144", 58537), 
("ch150", 6528), 
("kroA150", 26524), 
("kroB150", 26130), 
("pr152", 73682), 
("u159", 42080), 
("rat195", 2323), 
("d198", 15780), 
("kroA200", 29368), 
("kroB200", 29437), 
("ts225", 126643), 
("tsp225", 3919), 
("pr226", 80369), 
("gil262", 2378), 
("pr264", 49135), 
("a280", 2579), 
("pr299", 48191), 
("lin318", 42029), 
("linhp318", 41345), 
("rd400", 15281), 
("fl417", 11861), 
("pr439", 107217), 
("pcb442", 50778), 
("d493", 35002), 
("u574", 36905), 
("rat575", 6773), 
("p654", 34643), 
("d657", 48912), 
("u724", 41910), 
("rat783", 8806), 
("pr1002", 259045), 
("u1060", 224094), 
("vm1084", 239297), 
("pcb1173", 56892), 
("d1291", 50801), 
("rl1304", 252948), 
("rl1323", 270199), 
("nrw1379", 56638), 
("fl1400", 20127), 
("u1432", 152970), 
("fl1577", 22249), 
("d1655", 62128), 
("vm1748", 336556), 
("u1817", 57201), 
("rl1889", 316536), 
("d2103", 80450), 
("u2152", 64253), 
("u2319", 234256), 
("pr2392", 378032), 
("pcb3038", 137694), 
("fl3795", 28772), 
("fnl4461", 182566), 
("rl5915", 565530), 
("rl5934", 556045), 
("rl11849", 923368), 
("usa13509", 19982889), 
("brd14051", 469445), 
("d15112", 1573152), 
("d18512", 645488)
]


for tsp_filename, optimum in datasets:
    coords = read_tsp_file(f"./datasets/{tsp_filename}.tsp")
    graph = create_distance_matrix(coords)

    with open(f"./results/{tsp_filename}_twice.txt", "w") as file:
        try:
            twice_result, twice_cost, elapsed_time, memory_used, aprox = twice_around_the_tree(graph, optimum)
            
            file.write(f"{twice_cost}\n")
            for vertex in twice_result:
                file.write(f"{vertex}\t")
            file.write("\n")
            file.write(f"{elapsed_time}\n")
            file.write(f"{memory_used}\n")
            file.write(f"{aprox}\n")
        except TimeoutError:
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")


for tsp_filename, optimum in datasets:
    coords = read_tsp_file(f"./datasets/{tsp_filename}.tsp")
    graph = create_distance_matrix(coords)

    with open(f"./results/{tsp_filename}_christofides.txt", "w") as file:
        try:
            christofides_result, christofides_cost, elapsed_time, memory_used, aprox = christofides_tsp(graph, optimum)

            file.write(f"{christofides_cost}\n")
            for vertex in christofides_result:
                file.write(f"{vertex}\t")
            file.write("\n")
            file.write(f"{elapsed_time}\n")
            file.write(f"{memory_used}\n")
            file.write(f"{aprox}\n")
        except TimeoutError:
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")

for tsp_filename, optimum in datasets:
    coords = read_tsp_file(f"./datasets/{tsp_filename}.tsp")
    graph = create_distance_matrix(coords)
    with open(f"./results/{tsp_filename}_twice.txt", "r") as file:
        first_line = file.readline().strip()
        twice_cost = None if first_line=="NA" else float(first_line)
        
        second_line = file.readline().strip()
        if second_line != "NA":
            twice_tour = [float(val) for val in second_line.split('\t')]
        else:
            twice_tour = None 

    with open(f"./results/{tsp_filename}_christofides.txt", "r") as file:
        first_line = file.readline().strip()
        christofides_cost = None if first_line=="NA" else float(first_line)

        second_line = file.readline().strip()
        if second_line != "NA":
            christofides_tour = [float(val) for val in second_line.split('\t')]
        else:
            christofides_tour = None 

    with open(f"./results/{tsp_filename}_bnb.txt", "w") as file:
        try:
            if twice_cost:
                if christofides_cost:
                    if twice_cost < christofides_cost:
                        bnb_result, bnb_cost, elapsed_time, memory_used, aprox = bnb_tsp(graph, optimum, twice_tour, twice_cost)
                    else:
                        bnb_result, bnb_cost, elapsed_time, memory_used, aprox = bnb_tsp(graph, optimum, christofides_tour, christofides_cost)
                else:
                    bnb_result, bnb_cost, elapsed_time, memory_used, aprox = bnb_tsp(graph, optimum, twice_tour, twice_cost)
            elif christofides_cost:
                bnb_result, bnb_cost, elapsed_time, memory_used, aprox = bnb_tsp(graph, optimum, christofides_tour, christofides_cost)
            else:
                bnb_result, bnb_cost, elapsed_time, memory_used, aprox = bnb_tsp(graph, optimum)

            file.write(f"{bnb_cost}\n")
            for vertex in bnb_result:
                file.write(f"{vertex}\t")
            file.write("\n")
            file.write(f"{elapsed_time}\n")
            file.write(f"{memory_used}\n")
            file.write(f"{aprox}\n")
        except TimeoutError:
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
            file.write("NA\n")
