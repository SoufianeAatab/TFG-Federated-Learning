#include <assert.h>
#include <chrono>
#include <random>
#include <stdint.h>
#include <cmath>
#include <fstream>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef uint32_t bool32;

#define GLOROT_UNIFORM 0

std::mt19937 generator(42);
std::uniform_real_distribution<double> distribution;

struct memory_arena
{
    size_t Used;
    size_t Size;
    u8 *Base;
};
static memory_arena MemoryArena = {};

inline void *PushSize(memory_arena *Arena, size_t SizeToReserve)
{
    assert(Arena->Used + SizeToReserve <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += SizeToReserve;
    return (void *)Result;
}

void InitMemory(u32 Size)
{
    MemoryArena.Base = (u8 *)malloc(Size);
    MemoryArena.Size = Size;
    MemoryArena.Used = 0;
}

void printMemoryInfo()
{
    std::printf("\nMemory used %lu\nMemory available %lu\n\n", MemoryArena.Used, MemoryArena.Size - MemoryArena.Used);
}

#include "M4.cpp"
#include "src/activations.h"

void setRandomUniform(double low, double high)
{
    distribution = std::uniform_real_distribution<double>(low, high);
}

// Set Glorot uniform distribution for weights random intialization
void setGlorotUniform(u32 in, u32 out)
{
    double scale = sqrt(6.0f / ((f32)in + (f32)out));
    distribution = std::uniform_real_distribution<double>(-scale, scale);
}

struct Layer
{
    // Weight and bias matrices
    M w;
    M b;

    // Gradient and biases weights matrices
    M dw;
    M db;

    M vdw;
    M vdb;

    // Factory function to create a new layer object
    static Layer *create(u32 input_size, u32 output_size)
    {
        // Allocate memory for the layer on the memory arena
        Layer *l = (Layer *)PushSize(&MemoryArena, sizeof(Layer));

#if GLOROT_UNIFORM
        // Initialize the weight matrix with Glorot uniform distribution
        setGlorotUniform(input_size, output_size);
#else
        setRandomUniform(-0.5, 0.5);
#endif
        // Initialize the weight and bias matrices with random values
        l->w = M::rand(input_size, output_size);
        l->b = M::zeros(1, output_size);

        // Initialize the gradient matrices to zero
        l->dw = M::zeros(input_size, output_size);
        l->db = M::zeros(1, output_size);

        // Initialize the momentum matrices to zero
        l->vdw = M::zeros(input_size, output_size);
        l->vdb = M::zeros(1, output_size);
        return l;
    }

    // Forward propagation function
    M forward(M x)
    {
        // Check that the input matrix has the correct number of columns, this saves us from stupid bugs.
        assert(x.cols == w.rows);
        M Out = M::zeros(x.rows, w.cols);
        for (u32 i = 0; i < w.cols; ++i)
        {
            f32 accum = 0;
            for (u32 j = 0; j < x.cols; ++j)
            {
                accum += x.data[j] * w.data[j * w.cols + i];
                // maybe slightly improve the performance ?
                // accum += x.data[j] * w.data[idx];
                // idx += w.cols;
            }
            Out.data[i] = accum + b.data[i];
        }
        return Out;
    }

    // Backward propagation function
    M backward(M grad)
    {
        // Check that the gradient matrix has the correct number of columns
        assert(grad.cols == w.cols);
        M out = M::zeros(1, w.rows);
        for (u32 k = 0; k < w.rows; ++k)
        {
            //f32 accum = 0;
            for (u32 l = 0; l < grad.cols; ++l)
            {
                //accum += grad.data[l] * w.data[k * w.cols + l];
                out.data[k] += grad.data[l] * w.data[k * w.cols + l];
            }
            //out.data[k] = accum;
        }
        return out;
    }

    // Function to reset the gradient matrices to zero
    void resetGradients()
    {
        this->dw = M::zeros(this->dw.rows, this->dw.cols);
        this->db = M::zeros(1, this->db.cols);
    }

    // Function to calculate the weights matrix gradients, using the input gradient and the activation function of the current layer
    M getDelta(M grads, M a)
    {
        assert(w.rows == a.cols);
        M out = M::zeros(w.rows, w.cols);
        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                f32 g = grads.data[j];
                out.data[i * w.cols + j] = g * a.data[i];
            }
        }
        return out;
    }

    void setDelta(M grads, M a)
    {
        assert(w.rows == a.cols);
        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                f32 g = grads.data[j];
                dw.data[i * w.cols + j] = g * a.data[i];
            }
            db.data[i] = grads.data[i];
        }

    }

    void UpdateWeights(f32 lr, u32 batchsize = 1)
    {
        // scale the learning rate by the batch size. By default, the batch size is set to 1.
        lr = lr * (1.0f / (f32)batchsize);

        // MOMENTUM
        vdw = vdw * 0.9f + dw * (1.0f-0.9f);  
        vdb = vdb * 0.9f + db * (1.0f-0.9f);  

        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                // Update weights
                //w.data[i * w.cols + j] -= lr * this->dw[i * this->dw.cols + j];
                w.data[i * w.cols + j] -= lr * this->vdw[i * this->vdw.cols + j];
            }
            // Update bias
            //b.data[i] -= lr * this->db[i];
            b.data[i] -= lr * this->vdb[i];
        }

    }
};

struct Size3D{
    i32 h;
    i32 w;
    i32 c;
};

Size3D get_output_from_kernel(u32 ih, u32 iw, u32 kh, u32 kw, u32 kc){
    Size3D output;
    output.c = kc;
    output.h = ih - kh + 1;
    output.w = iw - kw + 1;
    return output;
}

struct Conv2D{
    Size3D input_size;
    Size3D output_size;
    u32 numKernels;
    u32 stride;

    M4 kernels;
    M4 dkernels;

    M bias;
    M db;
    
    M4 vdkernels;
    M vdb;
    //M kernels[outputChannels][inputChannels];
    //M dkernels[outputChannels][inputChannels];

    // Factory function to create a new layer object
    // u32 channels, u32 h, u32 w   
    static Conv2D *Create(u32 h, u32 w, u32 c, u32 kh, u32 kw, u32 kc)
    {
        // Allocate memory for the layer on the memory arena
        Conv2D *l = (Conv2D *)PushSize(&MemoryArena, sizeof(Conv2D));

        setRandomUniform(-0.5, 0.5);

        l->input_size.c = c;
        l->input_size.h = h;
        l->input_size.w = w;

        l->numKernels = kc;
        
        l->output_size = get_output_from_kernel(h, w, kh, kw, kc);

        l->kernels = M4::rand(kh, kw, c, kc);
        l->dkernels = M4::zeros(kh, kw, c, kc);
        l->vdkernels = M4::zeros(kh, kw, c, kc);
        l->vdb = M::zeros(1, kc);
        
        l->bias = M::zeros(1, kc);
        l->db = M::zeros(1, kc);

        l->stride = 1;
        return l;
    }

    // CONVOLVE2D CORRECT (COMPARED WITH KERAS)
    M3 convolve2D(M3 input){
        assert(input_size.c > 0 && input.d3 == input_size.c  && input.d1 == input_size.h && input.d2 == input_size.w && "Input image mismatch layer");
        // TODO: add padding
        const Size3D output_size = this->output_size;
        const u32 stride = 1;

        const u32 input_channels = this->input_size.c;
        const u32 output_channels = this->output_size.c;

        M3 output = M3::zeros(output_size.h, output_size.w, output_size.c);
        for(u32 j=0;j<output_channels;++j){
            for (u32 h=0;h<output_size.h;++h) {
                for( u32 w=0;w<output_size.w;++w){
                    f32 s = 0;
                    for(u32 i=0;i<input_channels;++i){
                        u32 w_start = w * stride;
                        u32 h_start = h * stride;
                        for(u32 l=0;l<kernels.d1;++l) {
                            for(u32 m=0;m<kernels.d2;++m) {
                                // TODO: should check for borders?
                                    // To conv: perform element-wise multiplication of the slices of the prev (input) matrix
                                    // then sum up all the values from all channels, add the bias to the sum of convolutions for each output channel
                                s += input(l+h_start,m+w_start,i) * kernels(l,m,i,j);
                            }
                        }
                        //s += bias.data[j]; 
                    }
                    output.set(h,w,j, s);
                    //output[j].data[(output_size.w * h) + w] = s;
                }
            }
        }
        return output;
    }

    Size3D getInputSize() const {
        return this->input_size;
    }

    Size3D getOutputSize() const {
        return this->output_size;
    }

    u32 getLinearFlattenedSize() const {
        return this->output_size.c * this->output_size.h * this->output_size.w;
    }

    void backward_conv(M3 X, M3 dh){
        //std::printf("%d=%d, %d=%d, %d=%d, %d=%d\n", dh.d1, this->output_size.c, dh.d2, this->output_size.h, dh.d3, this->output_size.w, this->numKernels, dh.d1);
        assert(dh.d3 == this->numKernels && this->output_size.c == dh.d3 && dh.d1 == this->output_size.h && dh.d2 == this->output_size.w && "BACKWARD CONV ASSERT");
        const u32 input_h = this->input_size.h;
        const u32 input_w = this->input_size.w;

        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        const u32 k_h = input_h - output_h + 1;
        const u32 k_w = input_w - output_w + 1;

        M dx = M::zeros(input_h, input_w);

        u32 stride = 1;
        dkernels = M4::zeros(k_h, k_w, input_size.c, numKernels);
        for(u32 p=0;p<this->numKernels;++p){
            for (int c=0;c<this->input_size.c;++c){
                f32 bias = 0.0f;
                for (int i = 0; i < output_h; i++) {
                    for (int j = 0; j < output_w; j++) {
                        f32 grad_output_ij = dh(i, j,p);//[i * output_w + j];

                        for (int k = 0; k < k_h; k++) {
                            for (int l = 0; l < k_w; l++) {
                                //dx[(i+k) * input_w + (j + l)] += grad_output_ij * kernels(k,l,c,p);//[p][c][k * k_w + l];
                                f32 a = dkernels(k,l,c,p) + (grad_output_ij* X(i+k, j+l, c));
                                dkernels.set(k,l,c,p, a);
                                // dx[(i+k) * input_w + (j + l)] += grad_output_ij * kernels(k,l,c,p);//[p][c][k * k_w + l];
                                // f32 a = dkernels(k,l,c,p) + (grad_output_ij * X(c, i+k, j+l));
                                // dkernels.set(k,l,c,p, a);
                            }
                        }
                        bias += dh(i,j,p);
                    }
                }
                db.data[c] = bias;
            }
        }
    }

    void updateKernels(f32 lr){
        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        vdkernels = vdkernels * 0.9f + dkernels * (1.0f-0.9f);
        vdb = vdb * 0.9f + db * (1.0f-0.9f);
        
        kernels -= vdkernels * lr;
        bias -= vdb * lr;

        // kernels -= dkernels * lr;
        // bias -= db * lr;
        // for(u32 p=0;p<this->numKernels;++p){
        //     for (int c=0;c<this->input_size.c;++c){
        //         for (int i = 0; i < output_h; i++) {
        //             for (int j = 0; j < output_w; j++) {
        //                 f32 old = kernels(p,c,i,j);
        //                 f32 dk = dkernels(p,c,i,j);
              
        //                 kernels.set(p,c,i,j, old - dk * lr);
        //             }
        //         }
        //     }
        // }
    }
};

struct MaxPooling{
    Size3D kernelsize;
    Size3D inputsize;
    Size3D outputsize;
    M3 d;
    static MaxPooling* create(i32 h, i32 w, i32 channels, i32 kh, i32 kw,i32 kc){
        MaxPooling *l = (MaxPooling *)PushSize(&MemoryArena, sizeof(MaxPooling));

        assert(h > 0 && channels > 0 && w > 0);
        std::printf("HOLA\n, %d, %d, %d", channels, h, w);

        l->d = M3::zeros(channels, h, w);

        l->kernelsize.c = kc;
        l->kernelsize.h = kh;
        l->kernelsize.w = kw;

        l->inputsize.c = channels;
        l->inputsize.h = h;
        l->inputsize.w = w;
        
        l->outputsize.h = h/kh;
        l->outputsize.w = w/kw;
        l->outputsize.c = channels;

        setRandomUniform(-0.5, 0.5);

        return l;
    }

    u32 getLinearFlattenedSize(){
        return outputsize.h * outputsize.w * outputsize.c;
    }

    M3 forward(M3 x){
        assert(x.d3 == inputsize.c && x.d1 == inputsize.h && x.d2 == inputsize.w);
        u32 output_h = outputsize.h;
        u32 output_w = outputsize.w;

        d = M3::zeros(inputsize.h, inputsize.w,inputsize.c);
        M3 output = M3::zeros(output_h, output_w, x.d3);
        for(u32 i=0;i<inputsize.c;++i){
            for(u32 l=0;l<output_h;++l){
                for( u32 p=0;p<output_w;++p){
                    f32 m = x(l*kernelsize.h, p*kernelsize.w,i);
                    u32 d3 = i;
                    u32 d1 = l*kernelsize.h;
                    u32 d2 = p*kernelsize.w;
                    for(u32 j=0;j<kernelsize.h;++j){
                        for (u32 k=0;k<kernelsize.w;++k){
                            if(l*kernelsize.h+j < x.d1 && p*kernelsize.w+k < x.d2){
                                if (x(l*kernelsize.h+j, p*kernelsize.w+k,i) > m) {
                                    // ALERT: we are using kernelsize as stride in this implementation
                                    m = x(l*kernelsize.h+j, p*kernelsize.w+k,i);
                                    d3 = i;
                                    d1 = l*kernelsize.h+j;
                                    d2 = p*kernelsize.w+k;
                                } 
                            }
                        }
                    }
                    output.set(l, p, i, m);
                    d.set(d1,d2,d3, 1.0f);
                }
            }
        }

        return output;
    }

    void backward(M3 grad){
        u32 u=0;
        for(u32 i=0;i<d.d1;++i){
            for (u32 j=0;j<d.d2;++j){
                for(u32 k=0;k<d.d3;++k){
                    if(d(i,j,k) == 1.0f){
                        d.set(i,j,k, grad.data[u++]);
                    }
                }
            }
        }
        assert(u == grad.d1 * grad.d2 * grad.d3);
    }

    void backward(M grad){
        u32 u=0;
        for(u32 i=0;i<d.d1;++i){
            for (u32 j=0;j<d.d2;++j){
                for(u32 k=0;k<d.d3;++k){
                    if(d(i,j,k) == 1.0f){
                        d.set(i,j,k, grad.data[u++]);
                    }
                }
            }
        }
        assert(u == grad.rows * grad.cols);
    }
};


#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define PORT 65432
f32 *get_data(u32 *size)
{
    int server_fd;
    int new_socket;
    struct sockaddr_in address;
    int addrlen;
    int opt = 1;
    addrlen = sizeof(address);
    std::printf("Waiting for incoming connection...\n");
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                             (socklen_t *)&addrlen)) < 0)
    {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    std::printf("Connected by %s \n", inet_ntoa(address.sin_addr));
    // receive 4 bytes for length indicator
    int bytes_length_count = 0;
    int bytes_length_total = 0;

    uint32_t length_descriptor = 0;
    char *len_buffer = reinterpret_cast<char *>(&length_descriptor);

    while (bytes_length_total < 4)
    {
        bytes_length_count = recv(new_socket,
                                  &len_buffer[bytes_length_total],
                                  sizeof(uint32_t) - bytes_length_total,
                                  0);

        if (bytes_length_count == -1)
        {
            perror("recv");
        }
        else if (bytes_length_count == 0)
        {
            std::printf("Unexpected end of transmission.\n");
            close(server_fd);
            exit(EXIT_SUCCESS);
        }

        bytes_length_total += bytes_length_count;
    }

    // receive payload
    int bytes_payload_count = 0;
    int bytes_payload_total = 0;

    size_t data_size = length_descriptor * sizeof(f32);
    f32 *data = (f32*)PushSize(&MemoryArena, data_size);
    char *buffer = reinterpret_cast<char *>(&data[0]);

    std::printf("Want to receive %zu\n", data_size);
    while (bytes_payload_total < static_cast<int>(data_size))
    {
        bytes_payload_count = recv(new_socket,
                                   &buffer[bytes_payload_total],
                                   data_size - bytes_payload_total,
                                   0);

        if (bytes_payload_count == -1)
        {
            perror("recv");
        }
        else if (bytes_payload_count == 0)
        {
            std::printf("Unexpected end of transmission.\n");
            close(server_fd);
            exit(EXIT_SUCCESS);
        }
        bytes_payload_total += bytes_payload_count;
    }

    *size = length_descriptor;
    close(server_fd);
    return data;
}

void creditcardFraudAutoEncoder()
{// Reserve memory arena which we'll use
    InitMemory(1024 * 1024 * 128);

    u32 size = 0;
    f32 *X = get_data(&size);
    assert(size > 0);
    M X_train[227451];
    for (u32 i = 0; i < 227451; ++i)
    {
        X_train[i] = {X + (i * 29), 1, 29};
    }
    X = get_data(&size);
    assert(size > 0);
    M X_test[56962];
    for (u32 i = 0; i < 56962; ++i)
    {
        X_test[i] = {X + (i * 29), 1, 29};
    }

    Layer *l1 = Layer::create(29, 20);
    Layer *l2 = Layer::create(20, 14);
    Layer *l3 = Layer::create(14, 20);
    Layer *l4 = Layer::create(20, 29);

    u32 MemoryUsed = MemoryArena.Used;
    printMemoryInfo();

    // Hyperparameters
    u32 epochs = 100;
    f32 lr = 0.001;
    u32 m = 227451; // 227451
    u32 m_valid = 56962;
    u32 batchsize = 32;

    // BEGIN_TRAIN_LOOP(m-(data_size), epochs, lr, batch_size)
    for (u32 epoch = 0; epoch < epochs; ++epoch) 
    {   
        f32 error = 0;
        f32 valid_error = 0.0f;
        for (u32 j = 0; j < m_valid; ++j)
        {
            // forward propagation
            M a = Tanh(l1->forward(X_test[j]));
            M b = Tanh(l2->forward(a));
            M c = Tanh(l3->forward(b));
            M o = l4->forward(c);
            // calculate error
            valid_error += Mse(X_test[j], o);
        }

        for (u32 i = 0; i < m; i += batchsize)
        {
            l1->resetGradients();
            l2->resetGradients();
            l3->resetGradients();
            l4->resetGradients();

            for (u32 j = i; j < (i + batchsize) && j < m; ++j)
            {
                // forward propagation
                M a = Tanh(l1->forward(X_train[j]));
                M b = Tanh(l2->forward(a));
                M c = Tanh(l3->forward(b));
                M o = l4->forward(c);

                // Backward propagation
                M _d4 = MsePrime(X_train[j], o);
                M _d3 = l4->backward(_d4) * TanhPrime(c);
                M _d2 = l3->backward(_d3) * TanhPrime(b);
                M _d1 = l2->backward(_d2) * TanhPrime(a);

                // accumulate gradients
                l1->dw += l1->getDelta(_d1, X_train[j]);
                l1->db += _d1;

                l2->dw += l2->getDelta(_d2, a);
                l2->db += _d2;

                l3->dw += l3->getDelta(_d3, b);
                l3->db += _d3;
                
                l4->dw += l4->getDelta(_d4, c);
                l4->db += _d4;

                // calculate error
                error += Mse(X_train[j], o);
            }

            l1->UpdateWeights(lr, batchsize);
            l2->UpdateWeights(lr, batchsize);
            l3->UpdateWeights(lr, batchsize);
            l4->UpdateWeights(lr, batchsize);

            MemoryArena.Used = MemoryUsed;
        }
        std::printf("Epoch[%i/%i] - Batch size: %u - Training Loss: %f - Valid Loss: %f - Learing rate: %f\n", 
                epoch, epochs, batchsize, error / (f32)m, valid_error / (f32) m_valid ,lr);
    }

    X_train[0].print();
    M a = Tanh(l1->forward(X_train[0]));
    M b = Tanh(l2->forward(a));
    M c = Tanh(l3->forward(b));
    M o = l4->forward(c);
    o.print();

    std::printf("Error: %f \n", Mse(X_train[0], o));
    free(MemoryArena.Base);
}

void CNN(){
     InitMemory(1024*1024*512*2);

    #define EPOCHS 1000
    #define M_EXAMPLES 60000
    #define TEST_EXAMPLES 10000
    #define SIZE 28
    #define CONVNET 1
    #define POOLING 1

    u32 size = 0;
    f32* x = get_data(&size);
    f32* ys = get_data(&size);

    f32* xtests = get_data(&size);
    f32* ytests = get_data(&size);

    M3 xtest[TEST_EXAMPLES];
    M ytest[TEST_EXAMPLES];
    
    M3 input[M_EXAMPLES];
    M y[M_EXAMPLES];

    for(u32 i=0;i<M_EXAMPLES;++i){
        input[i] = M3(x + i*SIZE*SIZE, SIZE,SIZE,1);
        y[i] = M(ys + i*10, 1,10);
    }

    for(u32 i=0;i<TEST_EXAMPLES;++i){
        xtest[i] = M3(xtests + i*SIZE*SIZE, SIZE,SIZE,1);
        ytest[i] = M(ytests + i*10, 1,10);
    }

    f32 lr = 0.01;
    u32 epochs = EPOCHS;
    
    #if CONVNET
    Conv2D* cnv1 = Conv2D::Create(SIZE,SIZE,1, 3,3,8); 
    #if POOLING
    MaxPooling* pl = MaxPooling::create(cnv1->getOutputSize().h, cnv1->getOutputSize().w, cnv1->getOutputSize().c, 2,2,1);
    Layer* l0 = Layer::create(pl->getLinearFlattenedSize(), 100);
    std::printf("OUTPUT MAXPOOLING %d, WITHOUT MAXPOOLING %d \n",pl->getLinearFlattenedSize(),cnv1->getLinearFlattenedSize());
    #else
    Layer* l0 = Layer::create(cnv1->getLinearFlattenedSize(), 100);
    #endif
    Layer* l1 = Layer::create(100, 10);
    std::printf("MEMORY USED %zu\n", MemoryArena.Used);
    u32 usedMem = MemoryArena.Used;

    std::printf("START TRAINING\n");
    for(u32 epoch=0;epoch<epochs;++epoch){
        f32 error = 0.0f;
        f32 v_error = 0.0f;
        f32 accuracy = 0.0f;
        f32 v_accuracy = 0.0f;

        for(u32 i=0;i<M_EXAMPLES;++i){
            l0->resetGradients();
            l1->resetGradients();

            M3 aa = Sigmoid(cnv1->convolve2D(input[i]));
            #if POOLING
                M3 bb = pl->forward(aa);
                M flatten(bb.data, 1, bb.d1*bb.d2*bb.d3);
            #else
                M flatten(aa.data, 1, aa.d1*aa.d2*aa.d3);
            #endif
            M a = Sigmoid(l0->forward(flatten));
            M c = Softmax(l1->forward(a));
            M d2 = M::MatMul(CrossEntropyPrime(y[i], c), SoftmaxPrime(c));
            M d1 = l1->backward(d2) * SigmoidPrime(a);
            M d0 = l0->backward(d1);

            #if POOLING
                M3 dcnv = M3(d0.data, bb.d1, bb.d2, bb.d3);
                pl->backward(dcnv);
                M3 bpl = pl->d * SigmoidPrime(aa);
                cnv1->backward_conv(input[i], bpl);
            #else
                M3 unflatten(d0.data, aa.d1, aa.d2, aa.d3);
                M3 ott = unflatten * SigmoidPrime(aa);
                cnv1->backward_conv(input[i], ott);
            #endif


            l1->dw = l1->getDelta(d2, a);
            l1->db = d2;

            l0->dw = l0->getDelta(d1, flatten);
            l0->db = d1;

            cnv1->updateKernels(lr);

            l0->UpdateWeights(lr);
            l1->UpdateWeights(lr);

            accuracy = c.argmax() == y[i].argmax() ? accuracy + 1 : accuracy;

            error += CrossEntropy(y[i], c);
            MemoryArena.Used = usedMem;

            if(i%1000 == 0) std::printf("DONE %d/%d\n",i, M_EXAMPLES);
        }

        for(u32 i=0;i<TEST_EXAMPLES;++i){
            M3 aa = Sigmoid(cnv1->convolve2D(xtest[i]));
            #if POOLING
                M3 bb = pl->forward(aa);
                M flatten(bb.data, 1, bb.d1*bb.d2*bb.d3);
            #else
                M flatten(aa.data, 1, aa.d1*aa.d2*aa.d3);
            #endif            
            M a = Sigmoid(l0->forward(flatten));
            M o = Softmax(l1->forward(a));
            
            v_accuracy = o.argmax() == ytest[i].argmax() ? v_accuracy + 1 : v_accuracy;
            v_error += CrossEntropy(ytest[i], o);

            MemoryArena.Used = usedMem;
        }

        MemoryArena.Used = usedMem;
        std::printf("Epoch(%d/%d) Train Error: %f, Train Acc: %f, Valid Error: %f, Valid Acc: %f\n", 
        epoch, epochs, error/ (f32)M_EXAMPLES, accuracy/ (f32)M_EXAMPLES, v_error / (f32)TEST_EXAMPLES, v_accuracy / (f32)TEST_EXAMPLES);
    }
    #else
        Layer* l0 = Layer::create(SIZE * SIZE, 100);
        Layer* l2 = Layer::create(100, 10);
        u32 usedMem = MemoryArena.Used;

      for(u32 epoch=0;epoch<epochs;++epoch){
        f32 error = 0.0f;
        f32 v_error = 0.0f;
        f32 accuracy = 0.0f;
        f32 v_accuracy = 0.0f;

        for(u32 i=0;i<TEST_EXAMPLES;++i){
            M xt = M(xtest[i].data, 1, SIZE * SIZE);

            M a = Relu(l0->forward(xt));
            M c = Softmax(l2->forward(a));    
            
            v_accuracy = c.argmax() == ytest[i].argmax() ? v_accuracy + 1 : v_accuracy;

            v_error += CrossEntropy(ytest[i], c);

            MemoryArena.Used = usedMem;
        }
        for(u32 i=0;i<M_EXAMPLES;++i){

            M x = M(input[i].data, 1, SIZE * SIZE);
            M a = Relu(l0->forward(x));
            M c = Softmax(l2->forward(a));    
            
            M d2 = M::MatMul(CrossEntropyPrime(y[i], c), SoftmaxPrime(c));
            M d0 = l2->backward(d2) * ReluPrime(a);

            l2->dw = l2->getDelta(d2, a);
            l2->db = d2;

            l0->dw = l0->getDelta(d0, x);
            l0->db = d0;

            l0->UpdateWeights(lr);
            l2->UpdateWeights(lr);

            accuracy = c.argmax() == y[i].argmax() ? accuracy + 1 : accuracy;

            error += CrossEntropy(y[i], c);
            MemoryArena.Used = usedMem;
        }
        MemoryArena.Used = usedMem;
      if(epoch > 0 && epoch % 5 == 0){

        std::printf("\nEpoch(%d/%d) Train Error: %f, Train Acc: %f, Valid Error: %f, Valid Acc: %f", 
        epoch, epochs, error/ (f32)M_EXAMPLES, accuracy/ (f32)M_EXAMPLES, v_error / (f32)TEST_EXAMPLES, v_accuracy / (f32)TEST_EXAMPLES);
        }
    }
    #endif 
    
    std::printf("\nMemory used: %lu kb\n", MemoryArena.Used/1024);
}

#include <iostream>

int main() {
    InitMemory(1024*1024 * 5); // Reserve 1MB of continous memory

    Layer* l1 = Layer::create(1,64);
    Layer* l2 = Layer::create(64,64);
    Layer* l3 = Layer::create(64,1);

    M x[600] = {};
    M y[600] = {};

    u32 size = 0;
    f32* xs = get_data(&size);
    f32* ys = get_data(&size);
    for(u32 i=0;i<600;++i){
        x[i] = M(xs + i,1,1);
        y[i] = M(ys + i,1,1);
    }
    f32 lr = 0.1;
    u32 m = 600;
    u32 epochs = 100;

    u32 memoryCheckPoint = MemoryArena.Used;
    #define TRAINING 0
    #if TRAINING
    for(u32 i=0;i<epochs;++i){
        f32 error = 0.0f;
        for(u32 j=0;j<m;++j){
            l1->resetGradients();
            l2->resetGradients();
            l3->resetGradients();

            M a = Sigmoid(l1->forward(x[j]));
            M b = Sigmoid(l2->forward(a));
            M o = l3->forward(b);

            M e3 = MsePrime(y[j], o);
            M e2 = l3->backward(e3) * SigmoidPrime(b);
            M e1 = l2->backward(e2) * SigmoidPrime(a);

            l1->setDelta(e1, x[j]);
            l2->setDelta(e2, a);
            l3->setDelta(e3, b);

            // l3->dw = l3->getDelta(e3, b);
            // l3->db = e3;

            // l2->dw = l2->getDelta(e2, a);
            // l2->db = e2;

            // l1->dw = l1->getDelta(e1, x[j]);
            // l1->db = e1;

            l1->UpdateWeights(lr);
            l2->UpdateWeights(lr);
            l3->UpdateWeights(lr);

            error += Mse(y[j], o);
            MemoryArena.Used = memoryCheckPoint;
        }

        std::printf("Epochs[%d/%d] loss: %f\n", i, epochs, error / 600.0f);
    }

    l1->w.print();
    l2->w.print();
    l3->w.print();

    l1->b.print();
    l2->b.print();
    l3->b.print();

    l1->w.store("w1");
    l1->b.store("b1");

    l2->w.store("w2");
    l2->b.store("b2");

    l3->w.store("w3");
    l3->b.store("b3");

    #else

    l1->w.load("w1");
    l1->b.load("b1");

    l2->w.load("w2");
    l2->b.load("b2");

    l3->w.load("w3");
    l3->b.load("b3");

    u32 index = 0;
    while(std::cin >> index){
        M a = Sigmoid(l1->forward(x[index]));
        M b = Sigmoid(l2->forward(a));
        M o = l3->forward(b);
        o.print();
    }

    #endif


}
