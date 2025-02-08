#define MAX_SEQ_LENGTH 512
#define EMBED_DIM 512
#define NUM_HEADS 8
#define FF_DIM 2048
#define VOCAB_SIZE 50000

typedef struct {

    float* weights;
    float* bias;
    int in_features;
    int out_features;

} Linear;

//nn.linear: affine linear transformation


typedef struct {

    Linear* query;
    Linear* key;
    Linear* value;
    Linear* output;
    int num_heads;
    int head_dim;
    
} MultiHeadAttention;

//MultiHead(Q,K,V) = Concat(head1, head2, ... headh)W^o
//where head_i = attention(QW^Q, KW^K, VW^V)


typedef struct {

    Linear* ff1;
    Linear* ff2; 

} FeedForward;

typedef struct {

    float* norm_weights;
    float* norm_bias;
    int dim;

} LayerNorm;

//LayerNorm: apply layer normalization to a mini batch of inputs
//speeds up training even with simple SGD



typedef struct {

    MultiHeadAttention* attention;
    FeedForward* feedforward;
    LayerNorm* norm1;
    LayerNorm* norm2;

}   TransformerEncoderLayer;

