# From auxeno on Kaggle
# https://www.kaggle.com/code/auxeno/set-transformers-tutorial-dl?scriptVersionId=191198070

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):
    "Calculates self-attention between input elements."
    def __init__(self, input_dim, projection_dim):
        super().__init__()
        
        # Projects each input element to Q K and V vectors
        self.query_encoder = nn.Linear(input_dim, projection_dim)
        self.key_encoder   = nn.Linear(input_dim, projection_dim)
        self.value_encoder = nn.Linear(input_dim, projection_dim)
        
        # Used for attention score calculation
        self.projection_dim = projection_dim
        
    def forward(self, x):
        # Get batch size and number of elements
        batch_size, num_elements, input_dim = x.shape
        
        # Reshape so each element is encoded
        x = x.view(batch_size * num_elements, input_dim)
        
        # Query key and value encodings
        q = self.query_encoder(x).view(batch_size, num_elements, -1)
        k = self.key_encoder(x).view(batch_size, num_elements, -1)
        v = self.value_encoder(x).view(batch_size, num_elements, -1)
        
        # Dot product of query and key matrices
        qk = torch.bmm(q, k.transpose(1,2))
        
        # Normalised element-wise attention scores
        attention_scores = torch.softmax(qk / math.sqrt(self.projection_dim), dim=-1)
        
        # Multiply by values
        return torch.bmm(attention_scores, v)
    
class MultiHeadAttention(nn.Module):
    "Concatenates the output of several self-attention blocks."
    def __init__(self, input_dim, projection_dim, num_heads):
        super().__init__()
        
        # Multiple self-attention blocks
        self.heads = nn.ModuleList(
            [SelfAttention(input_dim, projection_dim) for _ in range(num_heads)]
        )
        
    def forward(self, x):
        # Concatenate outputs of self-attention heads
        return torch.cat([head(x) for head in self.heads], dim=-1)

class FeedForward(nn.Module):
    "Element-wise encoding with an MLP."
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        # Get batch size and number of elements as these may vary
        batch_size, num_elements, element_dim = x.shape
        
        # Reshape so each element is encoded by encoder
        x = x.view(batch_size * num_elements, element_dim)
        
        # Encode each element 
        x = self.encoder(x)
        
        # Reshape back
        x = x.view(batch_size, num_elements, -1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        # Calculate projection dim
        projection_dim = input_dim // num_heads
        
        # Multi-head attention
        self.multi_head_attention = MultiHeadAttention(input_dim, projection_dim, num_heads)
        
        # Feed forward layer
        self.feed_forward = FeedForward(projection_dim * num_heads, input_dim)
        
        # Layer norm layers
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(input_dim), nn.LayerNorm(projection_dim*num_heads)]
        )
        
    def forward(self, x):
        # Pre-layer normalisation
        x_normalized = self.layer_norms[0](x)
        
        # Multi-head attention
        x_att = self.multi_head_attention(x_normalized)
        
        # Additive residual connection
        x = x_att + x
        
        # Pre-layer normalisation
        x_normalized = self.layer_norms[1](x)
        
        # Feed forward output
        x_ff = self.feed_forward(x_normalized)
        
        # Additive residual connection
        x = x_ff + x
        
        # Output
        return x
        
class SeededSelfAttention(nn.Module):
    "Calculates self-attention between input elements."
    def __init__(self, input_dim, projection_dim):
        super().__init__()
        
        # Projects each input element to Q K and V vectors
        self.key_encoder   = nn.Linear(input_dim, projection_dim)
        self.value_encoder = nn.Linear(input_dim, projection_dim)
        
        # Used for attention score calculation
        self.projection_dim = projection_dim
        
    def forward(self, x, seed_vectors):
        # Get batch size and number of elements
        batch_size, num_elements, input_dim = x.shape
        
        # Reshape so each element is encoded
        x = x.view(batch_size * num_elements, input_dim)
        
        # Query key and value encodings
        q = seed_vectors.expand(batch_size, -1, self.projection_dim)
        k = self.key_encoder(x).view(batch_size, num_elements, self.projection_dim)
        v = self.value_encoder(x).view(batch_size, num_elements, self.projection_dim)
        
        # Dot product of query and key matrices
        qk = torch.bmm(q, k.transpose(1,2))
        
        # Normalised element-wise attention scores
        attention_scores = torch.softmax(qk / math.sqrt(self.projection_dim), dim=-1)
        
        # Multiply by values
        return torch.bmm(attention_scores, v)
    
class PoolMultiHeadAttention(nn.Module):
    "Concatenates the output of several self-attention blocks."
    def __init__(self, input_dim, projection_dim, num_heads, num_seed_vectors):
        super().__init__()
        
        # Multiple self-attention blocks
        self.heads = nn.ModuleList(
            [SeededSelfAttention(input_dim, projection_dim) for _ in range(num_heads)]
        )
        
        # Seed vectors
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, projection_dim))
        
    def forward(self, x):
        # Concatenate outputs of self-attention heads
        return torch.cat([head(x, self.seed_vectors) for head in self.heads], dim=-1)
    
class SimpleFeedForward(nn.Module):
    "Simple MLP encoder."
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class TransformerPoolingEncoder(nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        # Calculate projection dim
        projection_dim = input_dim // num_heads
        
        # Multi-head attention
        self.multi_head_attention = PoolMultiHeadAttention(input_dim, projection_dim, num_heads, 1)
        
        # Get seed vectors
        self.seed_vectors = self.multi_head_attention.seed_vectors
        
        # Feed forward layer
        self.feed_forward = SimpleFeedForward(projection_dim * num_heads, input_dim)
        
        # Layer norm layers
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(input_dim), nn.LayerNorm(projection_dim*num_heads)]
        )
        
    def forward(self, x):
        # Pre-layer normalisation
        x_normalized = self.layer_norms[0](x)
        
        # Multi-head attention
        x_att = self.multi_head_attention(x_normalized)
        
        # Get seed vectors and repeat for skip connection
        x_seed = self.seed_vectors.repeat(1, 1, len(self.multi_head_attention.heads))
        
        # Additive residual connection
        x = x_att + x_seed
        
        # Pre-layer normalisation
        x_normalized = self.layer_norms[1](x.squeeze(1))
        
        # Feed forward output
        x_ff = self.feed_forward(x_normalized)
        
        # Additive residual connection
        x = x_ff + x.squeeze(1)
        
        # Output
        return x


class SetTransformer(nn.Module):
    "Full Set Transformer architecture. Encodes and pools elements."
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.transformer_encoder_1 = TransformerEncoder(input_dim, num_heads)
        self.transformer_encoder_2 = TransformerEncoder(input_dim, num_heads)
        self.transformer_pool = TransformerPoolingEncoder(input_dim, num_heads)
        
    def forward(self, x):
        x = self.transformer_encoder_1(x)
        x = self.transformer_encoder_2(x)
        return self.transformer_pool(x)
    

if __name__ == "__main__":
    # Example usage
    batch_size = 2 # Just batching, ignore
    num_elements = 300 # Number of tokens inputted
    input_dim = 9 # Dimension of each token
    num_heads = 4 # Number of heads in multi-head attention

    # Sample input
    # inputs = torch.randn((batch_size, num_elements, input_dim))
    inputs = tuple(np.random.randn(num_elements + i, input_dim) for i in range(batch_size))
    print(inputs)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    # Set Transformer
    set_transformer = SetTransformer(input_dim, num_heads)

    # Example usage
    output = set_transformer(inputs)
    print(output.shape)