import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, dim_model, vocab_size):
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim_model) #maps input to dim-model length vector
    
    def forward(self, x): #x: tensor of token indices
        return self.embedding(x) * math.sqrt(self.dim_model)
        #embedding layer looks up the vector for each token index
        #math.sqrt scales the embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, dim_model, seq_len, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, dim_model) #matrix of zeros of shape(seq_len, dim_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #vector of shape (seq_len)
                         #unsqueeze add an extra dimension, axis specified by the parameter 
        #vector of shape(dim_model)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))

        #apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        #apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) 

        #reshape by adding dimension on axis 0
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
"""

    PE(pos, 2i) = sin(pos/(10000^(2i/dim_model)))

        value of position encoding at even indices

    PE(pos, 2i+1) = cos(pos/(10000^(2i/dim_model)))

        value of position encoding at odd indices

    dim_model is the dimension of embeddings 
"""


class LayerNormalization(nn.Module):

    def __init__(self, features, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

"""

    layer normalization:

        compute mean and variance:
            /mu = 1/H /sum x_i,

            /sigma^2 = 1/H /sum (x_i - /mu)^2,

            where H is the number of features. This is just mean & variance formula

        
        normalize:

            x^_i = (x_i - \mu) / sqrt(/sigma^2 + \epsilon)

        scale and shift:

            y_i = /gamma x^_i + /betea
            
            apply learned parameters \gamma and \beta to normalized output 

"""

class FeedForwardBlock(nn.Module):

    
    def __init__(self, dim_model, dim_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_ff, dim_model)

    def forward(self, x):

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


"""

    fully connected layer used both in encoder and decoder

    --two linaer transformations and a RELU in between

        x1 = w_1 x + bi
        x2 = RELU(x1)
        y = w_w x2 + b2



"""

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim_model, h, dropout):
        super().__init__()
        self.dim_model = dim_model
        self.h = h

        assert dim_model % h == 0

        self.dim_k = dim_model
        self.w_q = nn.Linear(dim_model, dim_model, bias = False)
        self.w_k = nn.Linear(dim_model, dim_model, bias = False)
        self.w_v = nn.Linear(dim_model, dim_model, bias = False)
        self.w_o = nn.Linear(dim_model, dim_model, bias = False)
        self.dropout = nn.Dropout(dropout)

    
    @staticmethod
    def attention(query, key, value, mask, dropout):
        dim_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) /math.sqrt(dim_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dim_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dim_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dim_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.dim_k)

        return self.w_o(x)
    
'''

    input vectors:

    X = [x1,x2, .... xn]

    query, key, value:
     Q = XWQ, K = XWK, V = XWV

    
    Attention (Q, K, V) = softmax(QK^T/sqrt(dim_k))V

    head_i = Attention(Q_i, K_i, V_i)

    MultiHead(Q,K,V) = Concat(head1, head2, .... head_h) W^o

        where w^o is a learned weight matrix used to project the concateneated outputs back to desired dimension

        


'''