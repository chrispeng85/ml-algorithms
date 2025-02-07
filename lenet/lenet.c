
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "definitions.h"

LeNet* create_lenet(void) {

    LeNet* net = (LeNet*)malloc(sizeof(LeNet));
    if (!net) return NULL;

    net->conv1.weights = (float*)malloc(CONV1_FILTERS * INPUT_CHANNELS * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * sizeof(float)); //initialize weights
    net->conv1.bias = (float*)malloc(CONV1_FILTERS * sizeof(float));   //tied bias: one bias per kernel

    net->conv2.weights = (float*)malloc(CONV2_FILTERS * CONV1_FILTERS * CONV2_KERNEL_SIZE * CONV1_KERNEL_SIZE * sizeof(float));
    net->conv2.bias = (float*)malloc(CONV2_FILTERS * sizeof(float)); 

    int fc1_input_size = POOL2_OUTPUT_WIDTH * POOL2_OUTPUT_HEIGHT * CONV2_FILTERS;  //fully connected to the output of prev layer
    net->FC1.weights = (float*)malloc(fc1_input_size * FC1_NEURONS * sizeof(float)); //fully connected to each neuron in this layer
    net->FC1.bias = (float*)malloc(FC1_NEURONS * sizeof(float)); //

    net->FC2.weights = (float*)malloc(FC1_NEURONS * FC2_NEURONS *sizeof(float));
    net->FC2.bias = (float*)malloc(FC2_NEURONS * sizeof(float));

    net->FC3.weights = (float*)malloc(FC2_NEURONS * OUTPUT_NEURONS * sizeof(float));
    net->FC3.bias = (float*)malloc(OUTPUT_NEURONS * sizeof(float));

    net->conv1_output = (float*)malloc(CONV1_FILTERS * CONV1_OUTPUT_HEIGHT * CONV1_OUTPUT_WIDTH * sizeof(float));
    net->pool1_output = (float*)malloc(CONV1_FILTERS * POOL1_OUTPUT_HEIGHT * POOL1_OUTPUT_WIDTH * sizeof(float));
    net->conv2_output = (float*)malloc(CONV2_FILTERS * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH * sizeof(float));
    net->pool2_output = (float*)malloc(CONV2_FILTERS * POOL2_OUTPUT_HEIGHT * POOL2_OUTPUT_WIDTH * sizeof(float)); //output array init
    
    net->fc1_output = (float*)malloc(FC1_NEURONS * sizeof(float));
    net->fc2_output = (float*)malloc(FC2_NEURONS * sizeof(float));
    net->output = (float*)malloc(OUTPUT_NEURONS) * sizeof(float);

    return net;
}

void convolution_2d(float* input, float* kernel, float* output, int input_h, int input_w, int kernel_size, float bias) {


    for (int i = 0; i < input_h - kernel_size + 1; i++) { //loop through each kernel starting position on 0 axis

            for (int j = 0; j < input_w - kernel_size + 1; i++) { //loop through each kernel starting position on 1 axis

                float sum = 0.0f;
                for (int ki = 0; ki < kernel_size; k++) {//loop through kernel size

                    for (int kj = 0; kj < kernel_size; kj ++) { //loop through kernel size

                    sum += input[IDX2D(i+ki, j + kj, input_w)] * 
                    kernel[IDX2D(ki, kh, kernel_size)]; // kernel entry * input entry

                    }
                }
            output[IDX2D(i,j, input_w - kernel_size + 1)] = RELU(sum + bias); //pass through RELU function
            }
    }


}


void max_pooling_2d(float * input, float* output, int input_h, int input_w, int pool_size) {


    int output_h = input_h/pool_size;
    int output_h = input_w / pool_size;

    for (int i= 0; i < output_h; i ++) {

        for (int j = 0; j < output_w; j ++) {

                float max_val = -INFINITY;
                for (int pi = 0; pi < pool_size; p ++) {

                        for (int pj = 0; pj < pool_size; p ++) { //loop through the entire pool 

                            float val = input[IDX2D( i * pool_size + pi, j * pool_size + pj, input_w)]; //loop through each value of the pool
                            max_val = MAX(max_valm val); //find max value
                        }
                }

                output[IDX2D(i,j, output_w)] = max_val;


        }

    }

}


void fully_connected(float* input, float* weights, float* bias, float* output, int input_size, int output_size) {


    for (int i =0; i < output_size; i ++) {

        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {

                sum += input[j] * weights[IDX2D(i,j,input_size)]; //input * weight of neuron
        }
        output[i] = RELU(sum + bias[i]); //add bias and apply RELU

    }
 

}

void forward(LeNet* net, float* input) {

    //forward pass: conv1 -> RELU -> max_pool -> conv2 -> RELU -> max_pool -> flatten 
    // -> fc1 -> fc2 -> fc3

    //conv1 layer 
    for (int f = 0; f < CONV1_FILTERS; f ++) {

        convolution_2d(

            input,
            &net->conv1.weights[f * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE],
            &net->conv1_output[f * CONV1_OUTPUT_HEIGHT * CONV1_OUTPUT_WIDTH],
            INPUT_HEIGHT,
            INPUT_WIDTH,
            CONV1_KERNEL_SIZE,
            net->conv1.bias[f]

        );

    }

    //pool1 layer
    for (int f = 0; f < CONV1_FILTERS; f++) {


        max_pooling_2d(
            &net->conv1_output[f * CONV1_OUTPUT_HEIGHT * CONV1_OUTPUT_WIDTH],
            &net->pool1_output[f * POOL1_OUTPUT_HEIGHT * POOL1_OUTPUT_WIDTH],
            CONV1_OUTPUT_HEIGHT,
            CONV1_OUTPUT_WIDTH,
            POOL1_HEIGHT,
            POOL1_WIDTH
        );
    }

    //conv2 layer
    for (int f = 0; f < CONV2_FILTERS; f++) {

        for (int c = 0; c < CONV1_FILTERS; c++) {

            float* temp_output = (float*)malloc(CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH * sizeof(float));

            convolution_2d(
                &net->pool1_output[c * POOL1_OUTPUT_HEIGHT * POOL1_OUTPUT_WIDTH],
                &net->conv2.weights[IDX3D(f,c,0, CONV1_FILTERS, CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE)],
                temp_output,
                POOL1_OUTPUT_HEIGHT,
                POOL1_OUTPUT_WIDTH,
                CONV2_KERNEL_SIZE,
                0.0f
            );

            for (int i=0; i < CONV2_OUTPUT_HEIGHT * CONV@CONV1_OUTPUT_WIDTH; i ++) {

                    if (c == 0) {
                        net->conv2_output[f * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH + i] = temp_output[i];

                    }
                    else  {
                        net->conv2_output[f * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH + i] += temp_output[i];


                    }

            }

            free(temp_output);

        }

        //add bias and apply RELU
        for (int i = 0; i < CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH; i++) {

            net->conv2_output[f * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH + i] = RELU(net->conv2_output[f * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH + i] + net->conv2.bias[f]);


        }

    }

    //pool2 layer

    for (int f = 0 ; f < CONV2_FILTERS; f++) {


        max_pooling_2d(
            &net->conv2_output[f * CONV2_OUTPUT_HEIGHT * CONV2_OUTPUT_WIDTH],
            &net->pool2_output[f * POOL2_OUTPUT_HEIGHT * POOL2_OUTPUT_WIDTH],
            CONV2_OUTPUT_HEIGHT,
            CONV2_OUTPUT_WIDTH,
            POOL2_SIZE
        );

    }


    //flatten pool2 outut for fc layers
    int fc1_input_size = POOL2_OUTPUT_WIDTH * POOL2_OUTPUT_HEIGHT * CONV2_FILTERS;

    //fc1 
    fully_connected(net->pool2_output, net->FC1.weights[f * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE],
    &net->conv1_OUTPUT[f * CONV1_OUTPUT_HEIGHT * CONV1_OUTPUT_WIDTH],
    INPUT_HEIGHT,
    INPUT_WIDTH,
    CONV1_KERNEL_SIZE,
    net->conv1.bias[f]
    );

    //fc2
    fully_connected(net->pool2_output, net->FC1.weights, net->FC1.bias, net->fc1_output, fc1_input_size, FC1_NEURONS)

    //output layer
    fully_connected(net->fc2_output, net->FC3.weights, net->FC3.bias,
    net->output, FC2_NEURONS, OUTPUT_NEURONS);

}

void destroy_lenet(LeNet* net) {


    free(net->conv1.weights);
    free(net->conv1.bias);
    free(net->conv2.weights);
    free(net->conv2.bias);
    free(net->fc1.weights);
    free(net->fc1.bias);
    free(net->fc2.weights);
    free(net->fc2.bias);
    free(net->fc3.weights);
    free(net->fc3.bias);



}



