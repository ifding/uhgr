import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import sys
import scipy.io
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import torch
import random

from graph_sampler import GraphSampler

def prepare_node_task(graphs, args, max_nodes=0):
    
    print('Number of nodes: ', sum([G.number_of_nodes() for G in graphs]))
    print('Number of node features: ', sum([G.graph['feat_dim'] for G in graphs]))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    
    # minibatch
    dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=1, 
            shuffle=True,
            num_workers=args.num_workers)

    return train_dataset_loader, dataset_sampler.assign_feat_dim

def prepare_graph_task(graphs, args, val_idx, max_nodes=0):

    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ', 
            max([G.number_of_nodes() for G in graphs]), ', '
            "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
            "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim
    

def read_graphfile(datadir, dataname, max_nodes=None):
    ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    '''
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            #if val == 0:
            #    label_has_zero = True
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    #graph_labels = np.array(graph_labels)
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    #if label_has_zero:
    #    graph_labels += 1
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    
    return graphs


def read_graph(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Read graph data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    features, _ = preprocess_features(features)
    
    G = nx.from_dict_of_lists(graph)
    
    adj = nx.adjacency_matrix(G)
    
    #G.graph['feat_dim'] = features.shape[1]
    #G.graph['label'] = 0
    
    #features = features.tolist()
    
    for u in G.nodes():
            G.node[u]['feat'] = np.array(features[u])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    
    return adj, features, labels, [idx_train, idx_val, idx_test]


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#def mol_to_nx(mol):
#    G = nx.Graph()
#
#    for atom in mol.GetAtoms():
#        G.add_node(atom.GetIdx(),
#                   atomic_num=atom.GetAtomicNum(),
#                   formal_charge=atom.GetFormalCharge(),
#                   chiral_tag=atom.GetChiralTag(),
#                   hybridization=atom.GetHybridization(),
#                   num_explicit_hs=atom.GetNumExplicitHs(),
#                   is_aromatic=atom.GetIsAromatic())
#    for bond in mol.GetBonds():
#        G.add_edge(bond.GetBeginAtomIdx(),
#                   bond.GetEndAtomIdx(),
#                   bond_type=bond.GetBondType())
#    return G
#
#def read_moleculefile(datadir, dataname, max_nodes=None):
#    ''' Read data from molecule file from NCI
#
#    Returns:
#        List of networkx objects with graph and node labels
#    '''
#    prefix = os.path.join(datadir, 'kdd11', dataname)
#    data = scipy.io.loadmat(prefix + '.mat')
#    samples = data['train_data'] 
#    labels = data['train_label']
#    n = samples.shape[1]
#
#    X = []
#    sel = np.full((n,), True, dtype=bool)
#
#    for i in range(n):
#        str = samples[0][i][0][0][0][0][0][0].encode('ascii', 'ignore')
#        m = Chem.MolFromSmiles(str)
#        if m is None:
#            sel[i] = False
#        X.append(str)
#
#    X = np.asarray(X)[sel]
#    y = np.asarray(labels[0])[sel]
#
#    graphs=[]
#    node_labels=[]
#
#    for i in range(len(X)):
#        mol = Chem.MolFromSmiles(X[i])
#        G = mol_to_nx(mol)
#        if max_nodes is not None and G.number_of_nodes() > max_nodes:
#            continue
#        G.graph['label'] = 0 if int(y[i]) < 0 else int(y[i])
#        
#        for u in G.nodes():
#            node_labels.append(G.node[u]['atomic_num'])
#        graphs.append(G)
#
#    unique_node_labels = set(node_labels)
#    num_unique_node_labels = len(unique_node_labels)
#
#    num_pos = 0
#    num_neg = 0
#    
#
#    for i in range(len(graphs)):
#        for u in graphs[i].nodes():
#            node_label_one_hot = [0] * num_unique_node_labels
#            node_label = graphs[i].node[u]['atomic_num']
#            node_index = list(unique_node_labels).index(node_label)
#            node_label_one_hot[node_index] = 1
#            graphs[i].node[u]['label'] = node_label_one_hot   
# 
#        if int(graphs[i].graph['label']) > 0:
#            num_pos += 1
#        else:
#            num_neg += 1
#            
#    print('Total graph number: ', len(graphs))        
#    print('Positive graph number: ', num_pos)    
#    print('Negative graph number: ', num_neg)
#    
#    return graphs
    