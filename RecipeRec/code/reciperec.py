# dependencies 

from tqdm import tqdm
import numpy as np

## torch
import torch
import torch.nn as nn

##  gnn
from gnn import (GNN,
                 ScorePredictor,
                 node_drop)

## settransformer
from settransformer import SetTransformer

## utils
from utils import (norm,
                   )

## config
from constants import CONSTANTS

device = CONSTANTS.device

class Model(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.user_embedding = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(),
        )
        self.instr_embedding = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.ingredient_embedding = nn.Sequential(
            nn.Linear(46, 128),
            nn.ReLU()
        )
        self.recipe_combine2out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.graph = graph

        self.gnn = GNN(128, 128, 128, graph.etypes)
        self.pred = ScorePredictor()
        self.setTransformer_ = SetTransformer()

    def forward(self, positive_graph, negative_graph, blocks, input_features):
        user, instr, ingredient, ingredient_of_dst_recipe = input_features
        
        # major GNN
        user_major = self.user_embedding(user)
        user_major = norm(user_major)
        instr_major = self.instr_embedding(instr)
        instr_major = norm(instr_major)
        ingredient_major = self.ingredient_embedding(ingredient)
        ingredient_major = norm(ingredient_major)
        x = self.gnn(blocks, {'user': user_major, 'recipe': instr_major, 'ingredient': ingredient_major}, torch.Tensor([[0]]))
        # contrastive - 1
        user1 = node_drop(user, 0.1, self.training)
        instr1 = node_drop(instr, 0.1, self.training)
        ingredient1 = node_drop(ingredient, 0.1, self.training)

        user1 = self.user_embedding(user1)
        user1 = norm(user1)
        instr1 = self.instr_embedding(instr1)
        instr1 = norm(instr1)
        ingredient1 = self.ingredient_embedding(ingredient1)
        ingredient1 = norm(ingredient1)
        
        x1 = self.gnn(blocks, {'user': user1, 'recipe': instr1, 'ingredient': ingredient1}, torch.Tensor([[1]]))
        
        # contrastive - 2
        user2 = node_drop(user, 0.1, self.training)
        instr2 = node_drop(instr, 0.1, self.training)
        ingredient2 = node_drop(ingredient, 0.1, self.training)
        
        user2 = self.user_embedding(user2)
        user2 = norm(user2)
        instr2 = self.instr_embedding(instr2)
        instr2 = norm(instr2)
        ingredient2 = self.ingredient_embedding(ingredient2)
        ingredient2 = norm(ingredient2)
        
        x2 = self.gnn(blocks, {'user': user2, 'recipe': instr2, 'ingredient': ingredient2}, torch.Tensor([[1]]))

        # setTransformer
        all_ingre_emb_for_each_recipe = self.get_ingredient_neighbors_all_embeddings(blocks, blocks[1].dstdata['_ID']['recipe'], ingredient_of_dst_recipe).to(device)
        all_ingre_emb_for_each_recipe = norm(all_ingre_emb_for_each_recipe)
        total_ingre_emb = self.setTransformer_(all_ingre_emb_for_each_recipe) # 1
        total_ingre_emb = norm(total_ingre_emb)
        
        # scores
        x['recipe'] = self.recipe_combine2out(total_ingre_emb.add(x['recipe']))
        pos_score = self.pred(positive_graph, x)
        neg_score = self.pred(negative_graph, x)        

        return pos_score, neg_score, x1, x2

    # get ingre neighbors for each recipe nodes
    def get_recipe2ingreNeighbor_dict(self):
        max_length = 33
        out = {}
        neighbor_list = []
        ingre_length_list = []
        total_length_index_list = []
        total_ingre_neighbor_list = []
        total_length_index = 0
        total_length_index_list.append(total_length_index)
        for recipeNodeID in tqdm(range(self.graph.number_of_nodes('recipe'))):
            _, succs = self.graph.out_edges(recipeNodeID, etype='r-i')
            succs_list = list(set(succs.tolist()))
            total_ingre_neighbor_list.extend(succs_list)
            cur_length = len(succs_list)
            ingre_length_list.append(cur_length)
            
            total_length_index += cur_length
            total_length_index_list.append(total_length_index)
            while len(succs_list) < max_length:
                succs_list.append(77733)
            neighbor_list.append(succs_list)

        ingre_neighbor_tensor = torch.tensor(neighbor_list).to(device)
        ingre_length_tensor = torch.tensor(ingre_length_list).to(device)
        total_ingre_neighbor_tensor = torch.tensor(total_ingre_neighbor_list).to(device)
        return ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor

    # ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor = get_recipe2ingreNeighbor_dict()
    # print('ingre_neighbor_tensor: ', ingre_neighbor_tensor.shape)
    # print('ingre_length_tensor: ', ingre_length_tensor.shape)
    # print('total_length_index_list: ', len(total_length_index_list))
    # print('total_ingre_neighbor_tensor: ', total_ingre_neighbor_tensor.shape)

    def find(self, tensor, values):
        return torch.nonzero(tensor[..., None] == values)

    # example of find()
    # a = torch.tensor([0, 10, 20, 30])
    # b = torch.tensor([[ 0, 30, 20,  10, 77733],[ 0, 30, 20,  10, 77733]])
    # find(b, a)[:, 2]


    def get_ingredient_neighbors_all_embeddings(self, blocks, output_nodes, secondToLast_ingre):
        ingreNodeIDs = blocks[1].srcdata['_ID']['ingredient']
        recipeNodeIDs = output_nodes
        ingre_neighbor_tensor, ingre_length_tensor, total_length_index_list, total_ingre_neighbor_tensor = self.get_recipe2ingreNeighbor_dict()
        batch_ingre_neighbors = ingre_neighbor_tensor[recipeNodeIDs].to(device)
        batch_ingre_length = ingre_length_tensor[recipeNodeIDs]
        valid_batch_ingre_neighbors = self.find(batch_ingre_neighbors, ingreNodeIDs)[:, 2]
        
        # based on valid_batch_ingre_neighbors each row index
        _, valid_batch_ingre_length = torch.unique(self.find(batch_ingre_neighbors, ingreNodeIDs)[:, 0], return_counts=True)
        batch_sum_ingre_length = np.cumsum(valid_batch_ingre_length.cpu())
        
        total_ingre_emb = None
        for i in range(len(recipeNodeIDs)):
            if i == 0:
                recipeNode_ingres = valid_batch_ingre_neighbors[0:batch_sum_ingre_length[i]]
                a = secondToLast_ingre[recipeNode_ingres]
            else:
                recipeNode_ingres = valid_batch_ingre_neighbors[batch_sum_ingre_length[i-1]:batch_sum_ingre_length[i]]
                a = secondToLast_ingre[recipeNode_ingres]
        
            # all ingre instead of average
            a_rows = a.shape[0]
            a_columns = a.shape[1]
            max_rows = 5
            if a_rows < max_rows:
                # a = torch.cat([a, torch.zeros(max_rows-a_rows, a_columns).cuda()])
                a = torch.cat([a, torch.zeros(max_rows-a_rows, a_columns).to(device)])
            else:
                a = a[:max_rows, :]
            
            if total_ingre_emb == None:
                total_ingre_emb = a.unsqueeze(0)
            else:
                total_ingre_emb = torch.cat([total_ingre_emb,a.unsqueeze(0)], dim = 0)
                if torch.isnan(total_ingre_emb).any():
                    print('Error!')

        return total_ingre_emb
