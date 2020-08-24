import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from .model_utils import mask_upper_half_matrix
class SelfAttention(nn.Module):
    def __init__(self, k, mask=False, heads=10):
        super().__init__()
        self.k = k
        self.h = heads
        self.mask = mask

        #Compute Linear Transformations
        self.transform_queries = nn.Linear(k, k*heads, bias=False)
        self.transform_keys = nn.Linear(k, k*heads, bias=False)
        self.transform_values = nn.Linear(k, k*heads, bias=False)
        # Linear Transform that reduces dimensionality
        self.dimension_reduce = nn.Linear(k*heads, k)
        
        
    def forward(self,x): #input b x t x k
        b, t, k = x.size()
        h = self.h
        assert k == self.k, f'Different input dimension ({k}) form ({self.k})'

        #Transform: b,t,k => b,t,k*h 
        queries = self.transform_queries(x)
        keys = self.transform_keys(x)
        values = self.transform_values(x)
        #Separate heads from dimension
        # b,t,k*h => b,t,h,k
        queries = queries.view(b,t,h,k)
        keys = keys.view(b,t,h,k)
        values = values.view(b,t,h,k)
        #Matrix multiplication for each batch and each head. Hence we merge heads and batch in order tu use torch.bmm
        # Transpose b,t,h,k => b,h,t,k
        # Merge dim b,h,t,k => b*h,t,k
        queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
        keys = keys.transpose(1,2).contiguous().view(b*h,t,k)
        values = values.transpose(1,2).contiguous().view(b*h,t,k)
        # Scale
        queries = queries /(k**(0.25))
        keys = keys/(k**(0.25))
        
        #Use torch batch matrix mult, performs a matrix mult for each elemt of the batch.
        #Batch consist of batch sample and head
        # b*h,t,k x b*h,k,t => b*h,t,t 
        weights = torch.bmm(queries,keys.transpose(1,2))
        assert weights.size() == (b*h,t,t)

        if self.mask:
            # mask the upper diagonal, excluding the diagonal,
            # so each vector has no notion of the next vectors of the sequence
            mask_upper_half_matrix(weights, val=-np.inf, include_diagonal=False)

        soft_weights = F.softmax(weights, dim=2)
        #Multiply weights b*h,t,t x b*h,t,k => b*h,t,k 
        #(each row weight contains weights for each row vectors, row weights linear combination of rowvectors) 
        output = torch.bmm(soft_weights, values).view(b, h, t, k)
        #Merge the h heads on the k dimension
        #Transpose b,h,t,k =>  b,t,h,k 
        #View(merge) b,t,h,k => b,t,h*k 
        output = output.transpose(1,2).contiguous().view(b,t,h*k)
        #Reduce dimension b,t,h*k => b,t,k
        return self.dimension_reduce(output)

class TransformerModule(nn.Module):
    def __init__(self, k, heads, mask=False, hidden_layer_mult=4):
        super().__init__()
        self.self_attention = SelfAttention(k=k, heads=heads, mask=mask)
        self.layer_norm_1 = nn.LayerNorm(k)
        self.feed_forward = nn.Sequential(
            nn.Linear(k, hidden_layer_mult * k),
            nn.ReLU(),
            nn.Linear(hidden_layer_mult * k, k),
        )
        self.layer_norm_2 = nn.LayerNorm(k)
        
    def forward(self, x):
        attention = self.self_attention(x)
        norm = self.layer_norm_1(attention + x)
        ff_output = self.feed_forward(norm)
        output = self.layer_norm_2(ff_output+norm)
        return output