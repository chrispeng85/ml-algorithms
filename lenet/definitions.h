#define IDX2D(i,j, width) ((i)*(width) + j)
#define IDX3D(i,j,k,height, width) ((i)*(height)*(width) + (j)*(width) + (k))
#define IDX4D(i,j,k,l,depth, height, width) ((i)*(depth)*(height)*(width) + (j)*(height)*(width) + (k)*(width) + (l))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define RELU(x) (MAX(0.0f, (x)))


#define INPUT_WIDTH 32
#define INPUT_HEIGHT 32
#define INPUT_CHANNELS 1

#define CONV1_FILTERS 6   //num of filters in conv1, i.e. 6 channels / output size
#define CONV1_KERNEL_SIZE 5 
#define CONV1_OUTPUT_WIDTH ((INPUT_WIDTH - CONV1_KERNEL_SIZE + 1))
#define CONV1_OUTPUT_HEIGHT ((INPUT_HEIGHT - CONV1_KERNEL_SIZE + 1))

#define POOL1_SIZE 2 //pooling window size
#define POOL1_OUTPUT_WIDTH (CONV1_OUTPUT_WIDTH / POOL1_SIZE)
#define POOL1_OUTPUT_HEIGHT (CONV1_OUTPUT_HEIGHT / POOL1_SIZE)

#define CONV2_FILTERS 16 //num. of feature maps / output size
#define CONV2_KERNEL_SIZE 5 
#define CONV2_OUTPUT_WIDTH ((POOL1_OUTPUT_WIDTH - CONV2_KERNEL_SIZE + 1))
#define CONV2_OUTPUT_HEIGHT ((POOL1_OUTPUT_HEIGHT - CONV2_KERNEL_SIZE + 1))

#define POOL2_SIZE 2  
#define POOL2_OUTPUT_WIDTH (CONV2_OUTPUT_WIDTH / POOL2_SIZE)
#define POOL2_OUTPUT_HEIGHT (CONV2_OUTPUT_HEIGHT / POOL2_SIZE)

#define FC1_NEURONS 120
#define FC2_NEURONS 84
#define OUTPUT_NEURONS 10



typedef struct {

    float* weights;
    float* bias;

} ConvLayer;

typedef struct {

    float* weights;
    float* bias;

} FCLayer;


typedef struct {

    ConvLayer conv1;
    ConvLayer conv2;

    FCLayer FC1;
    FCLayer FC2;
    FCLayer FC3;

    float* conv1_output;
    float* pool1_output;
    float* conv2_output;
    float* pool2_output;
    float* fc1_output;
    float* fc2_output;
    float* output;

} LeNet;

LeNet* create_lenet(void);

void destroy_lenet(LeNet* net);
void forward(LeNet* net, float* input);
void convolution_2d(float* input, float* kernel, float* output, int input_h, int input_w, int kernel_size, float bias);
void max_pooling_2d(float* input, float* output, int input_h, int input_w, int pool_size);
void fully_connected(float* input, float* weights, float* bias, float* output, int input_size, int output_size);

