#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
from tensorflow.keras import backend as K
import time
import pickle
import numpy as np
import tensorflow as tf
import os
def get_weighted_edge(edges, folder_name):
    #Matches edge with weights
    edge_weight = load_edge_weights(folder_name)
    weight = []
    for row in edges:
        try:
            weight.append(edge_weight[tuple(map(float,row))])
        except KeyError:
            weight.append(edge_weight[tuple(map(float,row[::-1]))])
        
    edges = np.concatenate([edges, np.expand_dims(weight, axis = -1)], axis = 1)
    return edges
    
def precompute_dist_data_weighted(edge_index, num_nodes,folder_name,approximate=0,):
        #Precomputes weighted distances between every pair of nodes
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        edge_list = get_weighted_edge(edge_list, folder_name)
        graph.add_weighted_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        
        dists_dict = dict(nx.all_pairs_dijkstra_path_length(graph,cutoff=approximate if approximate>0 else None, weight = 'weight'))
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    dists_array[int(node_i), int(node_j)] = 1 / (dist + 1)
        return dists_array
def load_tg_dataset(triples_1, triples_2):
    #load the graph for position calculation
    graphs, features, edge_labels,_,_,_,_ = load_graphs(triples_1, triples_2)
    return nx_to_tg_data(graphs, features, edge_labels)
def load_graphs(triples_1,triples_2):
    #generate graph and edges for position calculation
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]
    graphs = []
    
    for triples in [triples_1+triples_2]:
        trip = np.delete(np.array(triples),1,1)
        #print(trip[:5])
        graph = nx.Graph()
        graph.add_edges_from(trip)
        graphs.append(graph)
    edge_labels = []
    features = []

    for g in graphs:
        n = g.number_of_nodes()
        feature = np.ones((n, 1))
        features.append(feature)

        label = np.zeros((n, n),dtype=int)
        for i, u in enumerate(g.nodes()):
            for j, v in enumerate(g.nodes()):
                if i>j:
                    label[i, j] = 1
        label = label
        edge_labels.append(label)


    return graphs, features, edge_labels, node_labels, idx_train, idx_val, idx_test
class Data:
    def __init__(self,x,edge_index):
        super(Data,self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.mask_link_positive = None 
        self.dists = None
        self.num_nodes = None
        self.count_nodes(x)
        self.anchor_set_indicator = None
        self.anchor_set = None
        self.anchor_size_num = None
        self.dists_max = None
        self.dists_argmax = None
        self.anchorset_id = None
        self.shape = x.shape
        self.dtype = object
        
    def count_nodes(self, x):
        self.num_nodes = x.shape[0]
        
def gen_mapping(kg):
    ent_map = {**kg.kg1.entities_id_dict, **kg.kg2.entities_id_dict}
    rel_map = {**kg.kg1.relations_id_dict, **kg.kg2.relations_id_dict}

    return ent_map, rel_map
def nx_to_tg_data(graphs, features, edge_labels=None):

    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        remove_edge = graph.selfloop_edges()
        graph.remove_edges_from(remove_edge)

        x = np.zeros(feature.shape)
        graph_nodes = list(map(int,list(graph.nodes)))
        graph_nodes = list(map(int,list(np.array(graph.nodes))))
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = K.variable(x, dtype = tf.dtypes.float32)

        edge_index = np.array(graph.edges)
        edge_index = tf.transpose(K.variable(edge_index, dtype = tf.dtypes.int64))#.permute(1,0)


        data = Data(x=x, edge_index=edge_index)
        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        data_list.append(data)
    return data_list
def get_tg_dataset(total_kg,dataset_name,title = 'split_cache_join_weighted',use_cache=True, remove_feature=False):
    #Precomputes distance between every pair of nodes for position calculation
    #data_name = folder_name.strip().split('/')[-2]
    data_name=dataset_name
    task = 'link'
    print("Inside")
    st=time.time()
    dataset = load_tg_dataset(total_kg.kg1.relation_triples_list,total_kg.kg2.relation_triples_list) #output data_lists
    print(time.time()-st)
    # precompute shortest path
    if not os.path.isdir('/RePS/datasets'):
        os.mkdir('/RePS/datasets')
    if not os.path.isdir('/RePS/datasets/cache'):
        os.mkdir('/RePS/datasets/cache')
    f1_name = '/RePS/datasets/cache/' + title + '_' + data_name + str(-1) + '_dists.dat'
    f2_name = '/RePS/datasets/cache/' + title + '_' + data_name + str(-1)+ '_dists_removed.dat'
    f3_name = '/RePS/datasets/cache/' + title + '_' + data_name + str(-1)+ '_links_train.dat'
    f4_name = '/RePS/datasets/cache/' + title + '_' + data_name + str(-1)+ '_links_val.dat'
    f5_name = '/RePS/datasets/cache/' + title + '_' + data_name + str(-1)+ '_links_test.dat'
    if use_cache and ((os.path.isfile(f2_name) and task=='link') or (os.path.isfile(f1_name) and task!='link')):
        with open(f1_name, 'rb') as f1:
            dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):

            data.dists = K.variable(dists_list[i], dtype = tf.dtypes.float32)

            if remove_feature:
                data.x = tf.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []
        for i, data in enumerate(dataset):
            dists = precompute_dist_data_weighted(data.edge_index.numpy(), data.num_nodes,folder_name,approximate=-1)
            dists_list.append(dists)
            data.dists = K.variable(dists, dtype = tf.dtypes.float32)

            if remove_feature:
                data.x = tf.ones((data.x.shape[0],1))
            data_list.append(data)

        with open(f1_name, 'wb') as f1,             open(f2_name, 'wb') as f2,             open(f3_name, 'wb') as f3,             open(f4_name, 'wb') as f4,             open(f5_name, 'wb') as f5:

            pickle.dump(dists_list, f1, protocol=4)
            pickle.dump(links_train_list, f3, protocol=4)
            pickle.dump(links_val_list, f4, protocol=4)
            pickle.dump(links_test_list, f5, protocol=4)
        print('Cache saved!')
#    self.data_list = data_list
    return data
def split_square(arr):
    tmp=arr.numpy()
    data_1 = tmp[0::2,0::2]
    data_2 = tmp[1::2,1::2]
    return data_1,data_2
def merge_square(dist_1, dist_2,i):
    if(i == 1):
        dist_1 = tf.math.scalar_mul(2,dist_1)
        dist_2 = tf.math.scalar_mul(2,tf.math.add(dist_1,tf.constant([1], dtype = tf.int64)))
    x_expanded = tf.expand_dims(dist_1, 1)                   
    y_expanded = tf.expand_dims(dist_2, 1)                   

    concatted = tf.concat([x_expanded, y_expanded], -1)  

    result = tf.reshape(concatted, [-1, dist_1.shape[0]*2]) 
    return result
def get_dist_max_one(anchorset_id, dist_1,dist_2, device):
    dist_max = np.zeros((dist_1.shape[0]+dist_1.shape[1],len(anchorset_id[0])))
    dist_argmax = np.zeros(dist_max.shape, dtype = int)
#     print(len(anchorset_id[0]))
#     for i in range(len(anchorset_id[0])):
#         print(len(anchorset_id[0][i]))
    for i in range(len(anchorset_id[0])):
        temp_id_1 = np.array(anchorset_id[0][i], dtype = int)//2
        temp_id_2 = (np.array(anchorset_id[1][i], dtype = int)-1)//2
#         dist_temp_1 = tf.gather(dist_1, temp_id_1, axis=1)
#         dist_temp_2 = tf.gather(dist_2, temp_id_2, axis=1)
        dist_temp_1=dist_1[:,temp_id_1]
        dist_temp_2=dist_2[:,temp_id_2]
#         dist_max_temp_1, dist_argmax_temp_1 = tf.math.reduce_min(dist_temp_1, axis=-1), tf.math.argmin(dist_temp_1, axis = -1)
#         dist_max_temp_2, dist_argmax_temp_2 = tf.math.reduce_min(dist_temp_2, axis=-1), tf.math.argmin(dist_temp_2, axis = -1)
        dist_max_temp_1, dist_argmax_temp_1 = np.amax(dist_temp_1, axis=-1), np.argmax(dist_temp_1, axis = -1)
        dist_max_temp_2, dist_argmax_temp_2 = np.amax(dist_temp_2, axis=-1), np.argmax(dist_temp_2, axis = -1)           
        dist_max_temp = merge_square(dist_max_temp_1,dist_max_temp_2,0)
        #print(dist_max_temp.shape)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = merge_square(temp_id_1[dist_argmax_temp_1]*2,temp_id_2[dist_argmax_temp_2]*2+1,0)
    return tf.convert_to_tensor(dist_max, dtype = tf.dtypes.float32), tf.convert_to_tensor(dist_argmax, dtype = tf.dtypes.int32)
def get_random_anchorset_custom_split(ele,c=0.5):
    n = len(ele)
    ele = np.array(ele)
    m = int(np.log2(n))
    copy = m//2

    anchorset_id_en = []
    anchorset_id_fr = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 3))+1
        for j in range(copy):
            anchor_idx = np.random.choice(len(ele),size=anchor_size,replace=False)
            anchor_en = ele[:,0][anchor_idx]
            anchor_fr = ele[:,1][anchor_idx]
            
            anchorset_id_en.append(anchor_en)
            anchorset_id_fr.append(anchor_fr)
    return np.concatenate(anchorset_id_en),np.concatenate(anchorset_id_fr)
def preselect_anchor(dits_1,dits_2, seeds,layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):
    anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    return get_random_anchorset_custom_split(seeds,c=1)
    
#     seed_path = '/ads_storage/deepak_chaurasiya/datasets/EN_DE_15K_V1/721_5fold/1/train_links_coded'
#     seeds = np.loadtxt(seed_path, dtype = int, delimiter = ' ')
#     st=time.time()
#     for i in range(anchor_size_num):
#         anchor_size = 2**(i+1)-1
                
#         anchors = np.random.choice(len(seeds), size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
#         print(anchors.shape)
#         data.anchor_set.append(anchors)
#     data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)
#     print(time.time()-st) 
"""
    anchorset_id_en, anchorset_id_fr = get_random_anchorset_custom_split(seeds,c=1)
    anchorset_id = [anchorset_id_en,anchorset_id_fr]
    dists_max,dists_argmax = get_dist_max_one(anchorset_id,dits_1,dits_2, device)
    return dists_max,dists_argmax,anchorset_id
"""
    

