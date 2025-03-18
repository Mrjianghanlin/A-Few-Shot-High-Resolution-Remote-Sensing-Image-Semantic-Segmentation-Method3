import numpy as np

def louvain_algorithm(graph, max_iterations=100):
    """
    :param graph: input graph represented as a dictionary of dictionaries
    :param max_iterations: maximum number of iterations to run Louvain algorithm
    :return: list of tuples representing communities in the graph
    """

    # initialize each node as its own community
    communities = {node: node for node in graph}

    # calculate initial modularity score
    modularity_score = calculate_modularity(graph, communities)

    # continue iterating until no more improvement can be made
    for i in range(max_iterations):
        # divide the graph into subgraphs
        subgraphs = divide_graph(graph, communities)

        # optimize each subgraph using gradient descent
        for subgraph in subgraphs:
            sub_communities = optimize_subgraph(subgraph)
            for node in sub_communities:
                communities[node] = sub_communities[node]

        # calculate new modularity score
        new_modularity_score = calculate_modularity(graph, communities)

        # stop if modularity score does not improve
        if new_modularity_score == modularity_score:
            break

        modularity_score = new_modularity_score

    # convert dictionary of communities to list of tuples
    return get_communities(communities)


def calculate_modularity(graph, communities):
    """
    :param graph: input graph represented as a dictionary of dictionaries
    :param communities: dictionary mapping nodes to their respective communities
    :return: modularity score of the graph
    """
    # calculate total weight of edges in graph
    m = sum([sum(graph[node].values()) for node in graph])

    # initialize empty dictionary to store intra- and inter-community weights
    intra_weights = {}
    inter_weights = {}

    # iterate over each node
    for node in graph:
        community = communities[node]

        # initialize intra- and inter-community weights for current community
        intra_weights.setdefault(community, 0)
        inter_weights.setdefault(community, 0)

        # iterate over each neighboring node
        for neighbor in graph[node]:
            neighbor_community = communities[neighbor]

            # calculate weight of edge between node and neighbor
            weight = graph[node][neighbor]

            # update intra- or inter-community weight accordingly
            if neighbor_community == community:
                intra_weights[community] += weight
            else:
                inter_weights[community] += weight

    # calculate modularity score using intra- and inter-community weights
    modularity_score = sum([(intra_weights[community]/m) - ((inter_weights[community]/m)**2) for community in set(communities.values())])

    return modularity_score


def divide_graph(graph, communities):
    """
    :param graph: input graph represented as a dictionary of dictionaries
    :param communities: dictionary mapping nodes to their respective communities
    :return: list of subgraphs
    """

    # initialize empty dictionary to store subgraphs
    subgraphs = {}

    # iterate over each node
    for node in graph:
        community = communities[node]

        # add node to corresponding subgraph
        subgraphs.setdefault(community, {})[node] = {}

        # iterate over each neighboring node
        for neighbor in graph[node]:
            neighbor_community = communities[neighbor]

            # add neighbor to corresponding subgraph if it belongs to same community as node
            if neighbor_community == community:
                subgraphs[community][node][neighbor] = graph[node][neighbor]

    return [subgraphs[community] for community in subgraphs]


def optimize_subgraph(subgraph):
    """
    :param subgraph: input subgraph represented as a dictionary of dictionaries
    :return: dictionary mapping nodes to their optimized communities
    """

    # initialize empty dictionary to store optimized communities
    communities = {}

    # number of nodes in subgraph
    n = len(subgraph)

    # calculate total weight of edges in subgraph
    m = sum([sum(subgraph[node].values()) for node in subgraph])

    # initialize adjacency matrix and degree vector
    adj_matrix = np.zeros((n,n))
    degree_vector = np.zeros(n)

    # populate adjacency matrix and degree vector
    index_to_node = {}
    node_to_index = {}
    index = 0
    for node in subgraph:
        index_to_node[index] = node
        node_to_index[node] = index

        degree = sum(subgraph[node].values())
        degree_vector[index] = degree

        # iterate over each neighboring node
        for neighbor in subgraph[node]:
            neighbor_index = node_to_index[neighbor]

            # set adjacency matrix entry to weight of edge between node and neighbor
            adj_matrix[index][neighbor_index] = subgraph[node][neighbor]

        index += 1

    # calculate modularity

  #matrix
  #modularity_matrix = adj_matrix - np.outer(degree_vector, degree_vector)/max_eigenvector_index

  # perform eigendecomposition on modularity matrix
  eigenvalues, eigenvectors = np.linalg.eigh(modularity_matrix)

  # find eigenvector corresponding to largest eigenvalue
  max_eigenvector_index = np.argmax(eigenvalues)
  max_eigenvector = eigenvectors[:, max_eigenvector_index]

  # initialize partition vector
  partition_vector = np.zeros(np)

  # assign nodes to communities based on sign of entries in largest eigenvector
  for i in range(len(max_eigenvector)):
    if max_eigenvector[i] >= 0:
        partition_vector[i] = 1
    else:
        partition_vector[i] = -1

  # convert partition vector to dictionary mapping nodes to their optimized communities
  for i in range(np):
    node = index_to_node[i]
    if partition_vector[i] == 1:
        communities[node] = node
    else:
        communities[node] = node + "_new"
    #return communities

def get_communities(communities):
   """
   :param communities: dictionary mapping nodes to their respective communities
   :return: list of tuples representing communities in the graph
   """
   # initialize empty dictionary to store communities
   community_dict = {}

   # iterate over each node
   for node in communities: 

    # add node to corresponding community
    community_dict.setdefault(communities[node], []).append(node)

   # convert dictionary of communities to list of tuples
   community_list = [(community, sorted(members)) for community, members in community_dict.items()]

   # sort list of communities by size
   community_list.sort(key=lambda x: len(x[1]), reverse=True)

   return community_list

def main():
    # read in graph from file
    with open('I:/论文资料/数据集/Louvain.txt', 'r') as f:
        edges = [line.strip().split() for line in f]
    graph = {}
    for node1, node2, weight in edges:
        graph.setdefault(node1, {})[node2] = int(weight)
        graph.setdefault(node2, {})[node1] = int(weight)

    # run Louvain algorithm on graph
    communities = louvain_algorithm(graph)

    # print out communities
    for i, (community, members) in enumerate(communities):
        print(f'Community {i+1}: {", ".join(members)}')