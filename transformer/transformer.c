#include <stdio.h>
#include <stdlib.h>
#include "definitions.h"



float* malloc_float_array(int size) { //helper function to dynamically allocate a float array of size 'size'



    float* arr = (float*) (float*)malloc(size * sizeof(float));
    
    if (!arr) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    return arr;
}

void free_linear(Linear* layer) { //free memory of a linear layer

    free(layer->weights);
    free(layer->bias);
    free(layer);
}

Linear* create_linear(int in_features, int out_features) { //create a linear layer and allocate memory
 
    Linear* layer = (Linear*)malloc(sizeof(Linear));
    layer->weights = malloc_float_array(in_features * out_features);
    layer->bias = malloc_float_array(out_features);
    layer->in_features = in_features;
    layer->out_features = out_features;


    //Xavier initialization
    
    float scale = sqrt(2.0 / (in_features + out_features));
    for (int i= 0; i < in_features * out_features; i++) {

            layer->weights[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    }

    memset(layer->bias, 0 ,out_features * sizeof(float));

    return layer;


}

LayerNorm* create_layer_norm(int dim) { //initialize a layer norm and allocate memory

    LayerNorm* norm = (LayerNorm*)malloc(sizeof(LayerNorm));

    norm->norm_weights = malloc_float_array(dim);
    norm->norm_bias = malloc_float_array(dim);
    norm->dim = dim;

    //initialize weight as 1 and bias as 0

    for (int i =0; i < dim; i++) {

        norm->norm_weights[i] = 1.0f;
        norm->norm_bias[i] = 0.0f;

    }

    return norm;

}


MultiHeadAttention* create_multihead_attention(int embed_dim, int num_heads) {


    MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    int head_dim = embed_dim / num_heads;

    mha->query = create_linear(embed_dim, embed_dim);
    mha->key = create_linear(embed_dim, embed_dim);
    mha->value = create_linear(embed_dim, embed_dim);
    mha->output = create_linear(embed_dim, embed_dim);
    mha->num_heads = num_heads;
    mha->head_dim = head_dim;

    return mha;
    
}

FeedForward* create_feedforward(int embed_dim, int ff_dim) {


    FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
    ff->ff1 = create_linear(embed_dim, ff_dim);
    ff->ff2 = create_linear(ff_dim, embed_dim);

    return ff;

}

TransformerEncoderLayer* create_transformer_encoder(int embed_dim, int num_heads, int ff_dim) {


       TransformerEncoderLayer* layer = (TransformerEncoderLayer*)malloc(sizeof(TransformerEncoderLayer));
       layer->attention = create_multihead_attention(embed_dim, num_heads);
       layer->feedforward = create_feedforward(embed_dim, ff_dim);
       layer->norm1 = create_layer_norm(embed_dim);
       layer->norm2 = create_layer_norm(embed_dim);

       return layer;


}

void layer_norm_forward(LayerNorm* norm, float* input, float* output, int seq_len) {


    for (int i= 0; i < seq_len; i++) {

        //calculate mean
        float mean = 0;
        for (int j = 0 ; j< norm->dim; j++) {

            mean += input[i * norm->dim + j];

        }
        mean /= norm->dim;


        //calculate variance

        float var = 0;
        for (int j = 0; j < norm->dim; j++) {

            float diff = input[i * norm-> dim + j] - mean;
            var += diff * diff;
        }

        var /= norm->dim;

        //normalize
        for (int j= 0; j < norm->dim; j++) {

            float normalized = (input[i * norm->dim + j] - mean)/sqrt(var + 1e-5);
            output[i * norm->dim + j] = normalized * norm->norm_weights[j] + norm->norm_bias[j];

        }
    }


}

void linear_forward(Linear* layer, float* input, float* output, int batch_size) {

    for (int i = 0; i < batch_size; i++) {

        for (int j = 0; j < layer->out_features; j++) {

            float sum = layer->bias[j];
            for (int j = 0; k < layer->in_features; k++) {

                sum += input[i * layer->in_features + k] * 
                layer->weights[k * layer->out_features + j];

            }

            output[i * layer->out_features + j] = sum;

        }
    }

}

void softmax(float* input, float* output, int size) {


    float max_val = input[0];
    for (int i = 1; i < size; i++) {

        if (input[i] > max_val) max_val = input[i];

    }

    float sum = 0;
    for (int i= 0; i < size; i++) {

        output[i] = exp(input[i] - max_val);
        sum += output[i];

    }

    for (int i =0 ; i< size; i++) {

        output[i] /= sum;

    }

}


void attention_forward(MultiHeadAttention* mha, float* input, float* output, int seq_len) {

    int batch_head_size = seq_len * mha->num_heads * mha->head_dim;

    float* q = malloc_float_array(batch_head_size);
    float* k = malloc_float_array(batch_head_size);
    float* v = malloc_float_array(batch_head_size);
    float* scores = malloc_float_array(seq_len * seq_len * mha->num_heads);

    linear_forward(mha->query, input, q, seq_len);
    linear_forward(mha->key, input, k, seq_len);
    linear_forward(mha->value, input, v, seq_len );

    for (int h = 0; h < mha->num_heads; h++) {

        for(int i = 0; i < seq_len; i++) {

            

        }

    }


}



/*

    vanishing/exploding gradients and xavier intialization

    O = f_L * f_(L-1) * ..... f_1(x)

    dW_o = dh(L) * dh(L-1) * .... dh(l+1) *dh(l) 

    the gradient is the product of L-l matrices M(L)... M(l+1) and the gradient vector v(l)

    probabilities: change into log space

    Xavier initialization: 

        fully connected layer:
            O_i = \sum w_ij * x_j
            assume 0 mean and \sigma^2 variance for the weight distribution
            and 0 mean and \gamma^2 for the inputs x_i

            E(O_i) = \sum E(w_ij x_j)
                   = \sum E(w_ij) E(x_j)
                   = 0

            var[O_i] = E[O_i^2] - (E(O_i))^2
                     = \sum E[w_ij^2 x_j^2] - 0
                     = \sum E[w_ij^2] E[x_j^2]
                     = n * \sigma^2 * \gamma*2

            we seek to satisfy 1/2 (n_in + n_out)\sigma^2 = 1

                forward pass: n_in * \sigma^2 = 1 to maintain gradient stability
                backpropagation: n_out * \sigma^2 = 1 to prevent gradient explosion / vanishing
                can't be satisfied at the same time (otherwise n_in == n_out)



            (a weaker condition than n_out* \sigma^2 = 1 && n_in * \sigma^2 = 1)

            
            \sigma = sqrt(2/(n_in + n_out))

            this will be the std.dev the xavier inialization samples from

    


*/


/*

    multi head attention:

        runs attention mechanism several times in parellel
        results are concatenated and linearly transformed into desired dimension

        three inputs: query, key and value

        encoder-decoder architecture:

            machine translation: input and outputs are of varying length and unaligned

            encoder takes input, decoder predicts subsequent tokens in the target sequence

            input text tokenized into word tokens,
            encoded via embedding layer

            encoder:
            
            outputs continuous representation (embedding) of the input text that is passed to decoder


*/

/*

    layer normalization:


    previously there was batch normalization:
        to reduce undesired covariate shift
        a_i = g_i/sigma_i (a_i - mu_i)
        mu_i = E[a_i]
        sigma_i = sqrt(E[(a_i - mu_i)^2])

    






*/