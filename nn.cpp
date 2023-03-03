#include <assert.h>
#include <chrono>
#include <random>
#include <stdint.h>
#include <cmath>

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

#define GLOROT_UNIFORM 1

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
    // std::cout << "want to reserve"<< SizeToReserve << "\n";
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

struct M
{
    f32 *data;
    u32 rows;
    u32 cols;

    // M (const M&) = delete;
    // M& operator= (const M&) = delete;

    M() = default;
    M(f32 *data, u32 rows, u32 cols) : data(data), rows(rows), cols(cols) {}

    static M ones(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            o.data[i] = 1.0f;
        }
        return o;
    }

    static M zeros(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    static M rand(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            o.data[i] = distribution(generator);
        }
        return o;
    }

    void shape()
    {
        std::printf("Shape: (%i, %i)\n", rows, cols);
    }

    void print()
    {
        this->shape();
        for (u32 i = 0; i < rows; ++i)
        {
            for (u32 j = 0; j < cols; ++j)
            {
                std::printf("%f ", data[i * cols + j]);
            }
            std::printf("\n");
        }
    }

    void transpose()
    {
        if (cols != 1)
        {
            u32 aux = cols;
            cols = rows;
            rows = aux;
        }
    }

    f32 &operator[](u32 idx)
    {
        return data[idx];
    }

    M operator-(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] - B.data[i];
        }
        return out;
    }

    M &operator*=(f32 v)
    {
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] = data[i] * v;
        }
        return *this;
    }

    M &operator+=(M v)
    {
        assert(this->cols == v.cols && this->rows == v.rows);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] = data[i] + v.data[i];
        }
        return *this;
    }

    M &operator-=(M v)
    {
        assert(this->cols == v.cols && this->rows == v.rows);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            data[i] -= v.data[i];
        }
        return *this;
    }

    M operator-()
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = -data[i];
        }
        return out;
    }

    M operator/(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * (1.0f / B.data[i]);
        }
        return out;
    }

    M operator*(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    M operator*(float B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * B;
        }
        return out;
    }

    void add(M b)
    {
        assert(rows == b.rows && cols == b.cols);
        for (u32 i = 0; i < b.rows * b.cols; ++i)
        {
            this->data[i] += b.data[i];
        }
    }

    void sub(M b)
    {
        assert(rows == b.rows && cols == b.cols);
        for (u32 i = 0; i < b.rows * b.cols; ++i)
        {
            this->data[i] -= b.data[i];
        }
    }

    float sum()
    {
        f32 s = 0.0f;
        for (u32 i = 0; i < rows * cols; ++i)
        {
            s += data[i];
        }
        return s;
    }

    float mean()
    {
        return (this->sum() / cols);
    }

    u32 argmax()
    {
        u32 imax = 0;
        for (u32 i = 0; i < rows * cols; ++i)
        {
            if (data[i] > data[imax])
            {
                imax = i;
            }
        }

        return imax;
    }

    M square()
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] * data[i];
        }
        return out;
    }

    static M MatMul(M A, M B)
    {
        assert(A.cols == B.rows);
        M Out = M::zeros(A.rows, B.cols);
        for (u32 i = 0; i < A.rows; i++)
        {
            for (u32 j = 0; j < B.cols; j++)
            {
                Out.data[i * B.cols + j] = 0;
                for (u32 k = 0; k < A.cols; k++)
                {
                    Out.data[i * B.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
            }
        }
        return Out;
    }

    static void MatMul_(M A, M B, M &Out)
    {
        for (u32 i = 0; i < A.rows; i++)
        {
            for (u32 j = 0; j < B.cols; j++)
            {
                Out.data[i * B.cols + j] = 0;
                for (u32 k = 0; k < A.cols; k++)
                {
                    Out.data[i * B.cols + j] += A.data[i * A.cols + k] * B.data[k * B.cols + j];
                }
            }
        }
    }
};

struct M3{
    f32* data;  
    u32 d1;
    u32 d2;
    u32 d3;

    M3() = default;

    M3(f32* data, u32 d1, u32 d2, u32 d3): data(data), d1(d1), d2(d2), d3(d3){

    }
    static M3 rand(u32 d1, u32 d2, u32 d3)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1 * d2 * d3 * sizeof(f32));
        M3 o(data, d1, d2, d3);
        for (u32 i = 0; i < d1*d2*d3; ++i)
        {
            o.data[i] = distribution(generator);
        }
        return o;
    }

    static M3 zeros(u32 d1, u32 d2, u32 d3)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1 * d2 * d3 * sizeof(f32));
        M3 o(data, d1, d2, d3);
        for (u32 i = 0; i < d1*d2*d3; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    f32 operator()(u32 i, u32 j, u32 k){
        u32 index = i * (d3*d2) + j*d3 + k;
        return data[index];
    }

    void set(u32 i, u32 j, u32 k, f32 val){
        u32 index = i * (d3*d2) + j*d3 + k;
        data[index] = val;
    }

    M3 operator*(M3 B)
    {
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    void print(){
        std::printf("SHAPE: %d, %d, %d\n",d1,d2,d3 );
        for(u32 i=0;i<d1;++i){
            std::printf("[\n");
            for(u32 j=0;j<d2;++j){
                std::printf("[");
                for(u32 k=0;k<d3;++k){
                    u32 index = i * (d3*d2) + j*d3 + k;
                    std::printf("%.9f, ",data[index]);
                }   
                std::printf("],\n");
            }
            std::printf("],\n");
        }
        std::printf("\n");
    }

    M operator()(u32 i){
        u32 index = i * (d2*d3);

        M o = M(data+index, d2,d3);
        return o;
    }

};

struct M4{
    f32* data;  
    u32 d1;
    u32 d2;
    u32 d3;
    u32 d4;

    M4() = default;
    M4(f32* data, u32 d1, u32 d2, u32 d3, u32 d4): data(data), d1(d1), d2(d2), d3(d3), d4(d4){}
    static M4 ones(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            o.data[i] = 1.0f;
        }
        return o;
    }

    static M4 zeros(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            o.data[i] = 0.0f;
        }
        return o;
    }

    static M4 rand(u32 d1, u32 d2, u32 d3, u32 d4)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, d1*d2*d3*d4 * sizeof(f32));
        M4 o(data, d1, d2, d3, d4);
        for (u32 i = 0; i < d1*d2*d3*d4; ++i)
        {
            o.data[i] = distribution(generator);
        }
        return o;
    }

    M4 &operator-=(M4 v)
    {
        assert(this->d1 == v.d1 && this->d2 == v.d2 && this->d3 == v.d3 && this->d4 == v.d4);
        for (u32 i = 0; i < d1 * d2*d3*d4; ++i)
        {
            data[i] -= v.data[i];
        }
        return *this;
    }

    M4 operator*(f32 lr)
    {
        M4 output = M4::zeros(d1,d2,d3,d4);
        for (u32 i = 0; i < d1 * d2*d3*d4; ++i)
        {
            output.data[i] = data[i] * lr;
        }
        return output;
    }

    f32 operator()(u32 l, u32 k, u32 j, u32 i){
        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        return data[index];
    }

    void set(u32 l, u32 k, u32 j, u32 i, f32 val){
        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        data[index] = val;
    }

    void add(u32 l, u32 k, u32 j, u32 i, f32 val){
        u32 index = l * (d4*d2*d3) + k * (d4*d3) + j*d4 + i;
        data[index] += val;
    }

    M operator()(u32 i, u32 j){
        u32 index = i * (d2*d3*d4) + j*(d3*d4);
        M o = M(data+index, d3, d4);
        return o;
    }

    M3 operator()(u32 i){
        u32 index = i * (d4*d2*d3);

        M3 o = M3(data+index, d2, d3,d4);
        return o;
    }

    void print(){
        std::printf("SHAPE: %d, %d, %d, %d\n",d1,d2,d3,d4 );

        for(u32 i=0;i<d1;++i){
            std::printf("[\n");
            for(u32 j=0;j<d2;++j){
                std::printf("[\n");
                for(u32 k=0;k<d3;++k){
                    std::printf("[");
                    for(u32 l=0;l<d4;++l){
                        u32 index = i * (d4*d2*d3) + j * (d4*d3) + k*d4 + l;
                        std::printf("%f, ",data[index]);
                    }   
                    std::printf("],\n");
                }   
                std::printf("],\n");
            }
            std::printf("]\n");
        }
    }

};

#include "src/activations.h"

struct Layer
{
    // Weight and bias matrices
    M w;
    M b;

    // Gradient and biases weights matrices
    M dw;
    M db;

    // Factory function to create a new layer object
    static Layer *create(u32 input_size, u32 output_size)
    {
        // Allocate memory for the layer on the memory arena
        Layer *l = (Layer *)PushSize(&MemoryArena, sizeof(Layer));

#if GLOROT_UNIFORM
        // Initialize the weight matrix with Glorot uniform distribution
        l->setGlorotUniform(input_size, output_size);
#else
        l->setRandomUniform(-0.05, 0.05);
#endif
        // Initialize the weight and bias matrices with random values
        l->w = M::rand(input_size, output_size);
        l->b = M::zeros(1, output_size);

        // Initialize the gradient matrices to zero
        l->dw = M::zeros(input_size, output_size);
        l->db = M::zeros(1, output_size);
        return l;
    }

    // Set Glorot uniform distribution for weights random intialization
    void setGlorotUniform(u32 in, u32 out)
    {
        double scale = sqrt(6.0f / ((f32)in + (f32)out));
        distribution = std::uniform_real_distribution<double>(-scale, scale);
    }

    void setRandomUniform(double low, double high)
    {
        distribution = std::uniform_real_distribution<double>(low, high);
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
            u32 idx = i;
            for (u32 j = 0; j < x.cols; ++j)
            {
                //accum += x.data[j] * w.data[j * w.cols + i];
                // maybe slightly improve the performance ?
                accum += x.data[j] * w.data[idx];
                idx += w.cols;
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

    void UpdateWeights(f32 lr, u32 batchsize = 1)
    {
        // scale the learning rate by the batch size. By default, the batch size is set to 1.
        lr = lr * (1.0f / (f32)batchsize);
        for (u32 i = 0; i < w.rows; ++i)
        {
            for (u32 j = 0; j < w.cols; ++j)
            {
                // Update weights
                w.data[i * w.cols + j] -= lr * this->dw[i * this->dw.cols + j];
            }
            // Update bias
            b.data[i] -= lr * this->db[i];
        }
    }
};

struct Size3D{
    u32 h;
    u32 w;
    u32 c;
};

Size3D get_output_from_kernel(Size3D input, Size3D kernel){
    return {input.h - kernel.h + 1, input.w - kernel.w + 1, kernel.c};
}

struct Conv2D{
    Size3D input_size;
    Size3D output_size;
    u32 numKernels;
    u32 stride;

    M4 kernels;
    M4 dkernels;
    //M kernels[outputChannels][inputChannels];
    //M dkernels[outputChannels][inputChannels];

    // Factory function to create a new layer object
    static Conv2D *Create(Size3D input_size, Size3D kernel_size)
    {
        // Allocate memory for the layer on the memory arena
        Conv2D *l = (Conv2D *)PushSize(&MemoryArena, sizeof(Conv2D));

        l->input_size = input_size;
        l->output_size = get_output_from_kernel(l->input_size, kernel_size);
        l->numKernels = kernel_size.c;
        l->kernels = M4::rand(kernel_size.h, kernel_size.w, input_size.c, kernel_size.c);
        l->dkernels = M4::zeros(kernel_size.h, kernel_size.w, input_size.c, kernel_size.c);

        l->stride = 1;
        return l;
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

    // To conv: perform element-wise multiplication of the slices of the prev (input) matrix
    // then sum up all the values from all channels, and finally add up the bias
    // Note: perform the previous calc for every filter in the current layer
    f32 conv_step(M a, u32 w, u32 h, M kernel){
        f32 s = 0.0f;
        for(u32 i=0;i<kernel.rows;++i) {
            for(u32 j=0;j<kernel.cols;++j) {
                // TODO: should check for borders?
                s += a.data[(h + i)*a.cols + (j+w)] * kernel.data[i*kernel.cols + j];
            }
        }

        return s;
    }

    // CONVOLVE2D CORRECT (COMPARED WITH KERAS)
    M3 convolve2D(M3 input){
        assert(input_size.c > 0 && input.d2 == input_size.h && input.d3 == input_size.w && "Input image mismatch layer");
        // TODO: add padding
        const Size3D output_size = this->output_size;
        const u32 stride = 1;

        //std::printf("Forward conv output size %d, %d\n", output_size.h, output_size.w);

        // Input channels/kernels, 32x32x3
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
                        for(u32 l=0;l<3;++l) {
                            for(u32 m=0;m<3;++m) {
                                // TODO: should check for borders?
                                s += input(i, l+h_start,m+w_start) * kernels(l,m,i,j);
                            }
                        }
                    }
                    output.set(h,w,j, s);
                    //output[j].data[(output_size.w * h) + w] = s;
                }
            }
        }
        return output;
    }

    void backward_conv(M3 X, M3 dh){
    assert(dh.d3 == this->numKernels && dh.d1 == this->output_size.h && dh.d2 == this->output_size.w);
        const u32 input_h = this->input_size.h;
        const u32 input_w = this->input_size.w;

        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        const u32 k_h = input_h - output_h + 1;
        const u32 k_w = input_w - output_w + 1;

        M dx = M::zeros(input_h, input_w);

        u32 stride = 1;
        dkernels = M4::zeros(k_w, k_w, input_size.c, numKernels);
        for(u32 p=0;p<this->numKernels;++p){
            for (int c=0;c<this->input_size.c;++c){
                for (int i = 0; i < output_h; i++) {
                    for (int j = 0; j < output_w; j++) {
                        f32 grad_output_ij = dh(i, j,p);//[i * output_w + j];
                        for (int k = 0; k < k_h; k++) {
                            for (int l = 0; l < k_w; l++) {
                                dx[(i+k) * input_w + (j + l)] += grad_output_ij * kernels(k,l,c,p);//[p][c][k * k_w + l];
                                f32 a = dkernels(k,l,c,p) + (grad_output_ij * X(c, i+k, j+l));
                                dkernels.set(k,l,c,p, a);
                                //dkernels(p,c,k,l)             += grad_output_ij * X(c, i+k, j+l);//[c][(i+k) * input_w + (j+l)];
                            }
                        }
                    }
                }
            }
        }
    }

    void updateKernels(f32 lr){
        const u32 output_w = this->output_size.w;
        const u32 output_h = this->output_size.h;

        kernels -= dkernels * lr;
        
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

// template <int inputChannels>
// struct MaxPooling{
//     Size3D kernelsize;
//     Size3D inputsize;
    
//     M* d;
//     static MaxPooling* create(Size3D inputSize, Size3D kernelsize){
//         MaxPooling *l = (MaxPooling *)PushSize(&MemoryArena, sizeof(MaxPooling));

//         l->d = (M*) PushSize(&MemoryArena, sizeof(M)*inputSize.c);

//         l->kernelsize = kernelsize;
//         l->inputsize = inputSize;
//         return l;
//     }

//     M* forward(M x[inputChannels]){
//         M* output = (M*) PushSize(&MemoryArena, sizeof(M)*inputChannels);
//         u32 output_h = inputsize.h/kernelsize.h;
//         u32 output_w = inputsize.w/kernelsize.w;
//         for(u32 i=0;i<inputChannels;++i){
//             output[i] = M::zeros(output_h, output_w);
//             for(u32 l=0;l<output_h;++l){
//                 for( u32 p=0;p<output_w;++p){
//                     f32 m = x[i].data[0];
//                     u32 index = 0;
//                     for(u32 j=0;j<kernelsize.h;++j){
//                         for (u32 k=0;k<kernelsize.w;++k){
//                             if(p*kernelsize.w+k < x[i].cols && l*kernelsize.h+j < x[i].rows){
//                                 if (x[i].data[l*kernelsize.h + p*kernelsize.w + j*x[i].cols  + k] > m)
//                                 {
//                                     m = x[i].data[l*kernelsize.h + p*kernelsize.w + j*x[i].cols + k];
//                                     index = l*kernelsize.h + p*kernelsize.w + j*x[i].cols  + k;
//                                 } 
//                             }
//                         }
//                     }
//                     d->data[index] = 1.0f;
//                     output[i].data[(l*output_w) + p] = m;
//                 }
//             }
//         }

//         d->print();
//         return output;
//     }
// };


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

int main()
{
    InitMemory(1024*1024*512*2);

    std::printf("MEMORY INTIALIZED\n");

    #define EPOCHS 10000
    #define M_EXAMPLES 1000
    #define TEST_EXAMPLES 797
    #define SIZE 8
    #define CONVNET 1

    u32 size = 0;
    f32* x = get_data(&size);
    M3 input[M_EXAMPLES];
    M input2[M_EXAMPLES];
    for(u32 i=0;i<M_EXAMPLES;++i){
        input[i] = M3(x + i*SIZE*SIZE, 1, SIZE,SIZE);
        input2[i] = M(x + i*SIZE*SIZE, 1, SIZE*SIZE);
    }
    
    f32* ys = get_data(&size);
    M y[M_EXAMPLES];
    for(u32 i=0;i<M_EXAMPLES;++i){
        y[i] = M(ys + i*10, 1,10);
    }

    f32* xtests = get_data(&size);
    M3 xtest[TEST_EXAMPLES];
    M xtest2[TEST_EXAMPLES];
    for(u32 i=0;i<TEST_EXAMPLES;++i){
        xtest[i] = M3(xtests + i*SIZE*SIZE, 1, SIZE,SIZE);
        xtest2[i] = M(xtests + i*SIZE*SIZE, 1, SIZE*SIZE);
    }
    
    f32* ytests = get_data(&size);
    M ytest[TEST_EXAMPLES];
    for(u32 i=0;i<TEST_EXAMPLES;++i){
        ytest[i] = M(ytests + i*10, 1,10);
    }

    f32 lr = 0.01;
    u32 epochs = EPOCHS;
    std::printf("START TRAINING\n");
    
    #if CONVNET
    Conv2D* cnv1 = Conv2D::Create({SIZE,SIZE,1},{3,3,32}); 
    std::printf("SIZE: %d\n", cnv1->getLinearFlattenedSize());
    Layer* l0 = Layer::create(cnv1->getLinearFlattenedSize(), 100);
    Layer* l1 = Layer::create(100, 10);
    //Layer* l2 = Layer::create(16, 10);

    f32 w[] = {
         0.01811075,

        -0.22047621,

         0.39133304,


       -0.45425707,

        -0.12132424,

        -0.36175254,

       -0.00107366,

         0.5620297 ,

         0.06265229
    };  
    //cnv1->kernels = M4(w, 3, 3, 1, 1);

    f32 l1w[] = {
        -1.28007650e-01, -2.19205722e-01, -1.58900052e-01,
         1.32453322e-01,  1.16311163e-01, -1.14881888e-01,
        -1.24667823e-01, -3.22343707e-02, -2.04592735e-01,
        -1.28578022e-01,
        2.92393655e-01, -3.19836020e-01, -1.69577792e-01,
        -2.81095833e-01, -2.50841230e-01,  2.75351375e-01,
        -3.11289400e-01,  1.63708955e-01, -2.47900337e-01,
         2.61865884e-01,
       -1.54715091e-01,  2.87669986e-01, -1.40497074e-01,
        -2.66147733e-01, -1.39693961e-01,  1.05472207e-02,
        -1.52990207e-01,  8.00245404e-02, -2.68996298e-01,
         1.55115753e-01,
       -5.32827377e-02,  2.11962074e-01,  1.72762126e-01,
        -3.61142397e-01,  6.83674812e-02,  7.82917142e-02,
         2.16739088e-01, -3.45993757e-01, -2.30607018e-01,
        -1.43070206e-01,
       -1.78656101e-01, -1.13937303e-01,  2.79924363e-01,
        -2.73034275e-01, -4.11961079e-02, -1.85511664e-01,
         2.16454536e-01, -3.52925509e-01, -3.30593050e-01,
        -2.53453523e-01,
        2.23061651e-01,  7.97308087e-02,  1.12403750e-01,
         1.09187037e-01,  6.53584898e-02, -2.49928504e-01,
         9.51654315e-02, -1.63660973e-01,  1.37838721e-03,
        -1.35758027e-01,
       -2.95112371e-01, -7.78616071e-02, -1.59642294e-01,
        -2.14611158e-01, -3.59166622e-01,  1.58911645e-02,
         3.10908705e-01,  3.26632649e-01,  3.92968357e-02,
         2.54140645e-01,
        2.04142302e-01, -1.78928718e-01, -3.12997401e-01,
         2.81572610e-01,  2.78133363e-01,  8.33083689e-02,
         1.10188812e-01,  4.37136889e-02,  3.11842263e-02,
        -9.08667445e-02,
       -1.68277502e-01, -3.03647518e-01,  6.74803257e-02,
        -2.51160085e-01, -7.86974430e-02,  2.01508671e-01,
        -3.37519318e-01,  3.58500510e-01, -2.47480571e-01,
        -2.28306651e-02,
       -2.75716245e-01, -2.21218035e-01, -2.13938236e-01,
         2.88266450e-01,  3.38028103e-01, -1.04075402e-01,
        -1.96368098e-02,  2.58489639e-01, -3.07655275e-01,
         9.20528471e-02,
       -9.69347358e-03, -7.50808716e-02,  1.44541651e-01,
        -2.86098480e-01, -1.49218127e-01, -8.03698897e-02,
        -1.88787326e-01,  2.00558394e-01,  6.03478551e-02,
         1.29476190e-01,
       -1.54030368e-01, -1.12082049e-01,  1.49244756e-01,
        -1.36355251e-01,  9.78658199e-02,  1.49488449e-04,
        -2.36767530e-01, -1.65234655e-01,  5.29852211e-02,
         3.56810361e-01,
       -3.47554684e-01, -6.03264272e-02, -1.87135026e-01,
        -2.47740179e-01, -2.14836895e-02,  3.42378646e-01,
         8.13631415e-02, -3.56429368e-01, -2.00764686e-01,
         6.13320470e-02,
        1.04408145e-01,  2.55502194e-01, -1.07804447e-01,
         3.10725242e-01, -1.91916451e-01, -8.71485472e-03,
        -3.16003382e-01, -1.58253610e-02,  3.26265186e-01,
         2.96122432e-02,
        3.01327735e-01, -9.23148692e-02, -6.46678209e-02,
        -7.56520331e-02, -2.03847289e-02, -3.13718200e-01,
         3.03086609e-01,  1.97797209e-01, -1.93804085e-01,
         1.12495720e-02,
        5.39077818e-02, -2.36931890e-01, -3.05149138e-01,
         4.35426831e-02, -2.70862490e-01, -2.51354516e-01,
         3.33012074e-01, -8.26079845e-02,  8.91436636e-02,
        -2.21777290e-01,
       -2.49126673e-01, -1.46749020e-01, -1.05434537e-01,
         3.52074802e-02, -6.72289133e-02, -3.37643743e-01,
        -5.17170429e-02, -2.91535556e-02,  1.61100775e-01,
         2.71005660e-01,
       -1.02996588e-01, -1.67282104e-01,  3.47996444e-01,
        -1.96794301e-01, -3.43312562e-01, -1.59927472e-01,
        -3.16172510e-01,  1.92128807e-01,  1.48906082e-01,
        -3.36319238e-01,
       -1.86794400e-01, -2.15406954e-01,  2.69073278e-01,
        -2.58115441e-01,  3.42212349e-01, -3.52989137e-01,
         3.54293257e-01, -3.10369253e-01,  1.85572952e-01,
        -3.36662024e-01,
        3.69140208e-02, -2.60214031e-01,  2.03639895e-01,
        -3.60198259e-01,  3.36535722e-01,  3.32322150e-01,
        -1.82839632e-02, -3.20235461e-01, -1.93757758e-01,
         7.33341277e-02,
       -1.08160317e-01, -1.12043470e-01,  4.68880832e-02,
        -2.32582048e-01, -2.73691446e-01,  2.84924597e-01,
         1.60495073e-01, -2.17559442e-01, -6.50054216e-03,
        -1.22247964e-01,
        1.50668412e-01,  1.43030673e-01, -4.66302931e-02,
         6.06492460e-02,  1.86451763e-01,  2.85112292e-01,
        -1.85821384e-01, -2.49671981e-01,  2.52585739e-01,
        -9.28188562e-02,
       -7.84060359e-03,  2.86419660e-01, -2.15958461e-01,
         3.01613063e-01, -1.91565216e-01,  2.91752070e-01,
         3.04199487e-01,  3.09268206e-01,  3.23797852e-01,
        -5.75291514e-02,
        1.16702110e-01,  1.41206950e-01, -5.44591248e-02,
         3.54511827e-01, -1.12548485e-01, -2.89758444e-02,
        -3.28590482e-01, -4.06547487e-02, -2.36206800e-01,
         2.05445558e-01,
        1.83267266e-01,  1.50170922e-03,  5.17060161e-02,
        -2.95081466e-01, -2.23703593e-01,  4.61525619e-02,
         1.39128000e-01, -3.45264256e-01,  4.10832167e-02,
        -2.23176360e-01,
       -2.20267415e-01, -1.19994819e-01, -2.21863747e-01,
        -2.32635945e-01,  2.35431045e-01,  1.28552258e-01,
         3.05459887e-01, -3.34879160e-02, -3.03248167e-01,
         1.03690952e-01,
        6.32165074e-02,  1.42986685e-01, -2.60309875e-01,
        -1.54826939e-01,  1.15662277e-01,  1.11587375e-01,
         7.13889003e-02,  1.04045033e-01,  1.00925297e-01,
        -1.14605427e-02,
       -5.93900979e-02,  3.15415472e-01, -1.25802800e-01,
        -6.92835748e-02,  1.67497188e-01,  1.84068710e-01,
         1.83549196e-01, -3.17370176e-01, -1.78091332e-01,
        -1.23011559e-01,
       -1.83028281e-02,  2.00938433e-01,  2.09325999e-01,
         2.83277184e-01, -4.35381234e-02, -1.84271634e-02,
        -1.64203957e-01,  1.32638216e-03,  1.91881329e-01,
         2.47538239e-01,
        1.99561268e-01, -2.28212386e-01,  1.78729892e-02,
         1.53174132e-01, -2.48630524e-01, -2.23883376e-01,
         6.15604222e-02, -2.17538521e-01, -2.62711465e-01,
        -1.28364474e-01,
        2.48340160e-01,  1.93155676e-01, -8.82413983e-03,
         6.99301660e-02, -1.73217431e-01,  3.27492625e-01,
         2.28378385e-01,  5.08687198e-02,  1.10150069e-01,
         1.64311439e-01,
        1.72499120e-02, -3.18811327e-01, -2.24930272e-01,
        -2.90882707e-01,  8.56738091e-02,  1.95489615e-01,
        -3.46374691e-01, -1.30055085e-01, -1.57186523e-01,
         2.15712041e-01,
       -1.31649792e-01, -2.07648143e-01,  8.05363357e-02,
        -6.70616031e-02,  1.98740035e-01, -1.02656484e-02,
         2.33342201e-01, -2.47967839e-01,  2.53239274e-03,
        -1.48087978e-01,
        1.35485053e-01,  1.65851265e-01,  2.40571827e-01,
         1.75702304e-01, -9.34783518e-02,  1.12397462e-01,
         1.21777356e-02,  3.10046881e-01, -2.08033040e-01,
        -1.39893889e-01,
       -2.15140715e-01, -2.19157040e-02,  2.09922582e-01,
         2.74526566e-01,  1.56198114e-01,  1.31547391e-01,
         3.07956308e-01,  9.05997157e-02, -1.80852517e-01,
        -3.38567644e-01,
        2.77283192e-02,  1.28311425e-01,  2.95827121e-01,
         1.13335490e-01,  1.14861995e-01, -3.06742877e-01,
        -1.49937123e-01, -2.63098240e-01,  1.42169446e-01,
        -1.27706438e-01
    };
    //l0->w = M(l1w, 36, 10);
    //l0->b = M::zeros(1, 10);
//    cnv1->kernels.print();


    std::printf("MEMORY USED %d\n", MemoryArena.Used);
    u32 usedMem = MemoryArena.Used;

    for(u32 epoch=0;epoch<epochs;++epoch){
        f32 error = 0.0f;
        f32 v_error = 0.0f;
        f32 accuracy = 0.0f;
        f32 v_accuracy = 0.0f;

        for(u32 i=0;i<TEST_EXAMPLES;++i){
            M3 aa = Sigmoid(cnv1->convolve2D(xtest[i]));
            M flatten(aa.data, 1, aa.d1*aa.d2*aa.d3);
            
            M a = Sigmoid(l0->forward(flatten));
            M o = Softmax(l1->forward(a));
            
            v_accuracy = o.argmax() == ytest[i].argmax() ? v_accuracy + 1 : v_accuracy;
            v_error += CrossEntropy(ytest[i], o);

            MemoryArena.Used = usedMem;
        }

        for(u32 i=0;i<M_EXAMPLES;++i){
            l0->resetGradients();
            M3 aa = Sigmoid(cnv1->convolve2D(input[i]));
            //a.flatten();
            M flatten(aa.data, 1, aa.d1*aa.d2*aa.d3);
            
            M a = Sigmoid(l0->forward(flatten));
            M c = Softmax(l1->forward(a));
            //M c = Softmax(l2->forward(a));    
            
            M d2 = M::MatMul(CrossEntropyPrime(y[i], c), SoftmaxPrime(c));
            M d1 = l1->backward(d2) * SigmoidPrime(a);
            M d0 = l0->backward(d1);
            // i could multiply here by Sigmoidprime. ¿and then unflatten?
            //M d0 = l0->backward(d0);

            l1->dw = l1->getDelta(d2, a);
            l1->db = d2;

            l0->dw = l0->getDelta(d1, flatten);
            l0->db = d1;

            M3 unflatten(d0.data, aa.d1, aa.d2, aa.d3);
            M3 ott = unflatten * SigmoidPrime(aa);
      
            //std::printf("iteration %d\n", i);
            cnv1->backward_conv(input[i], ott);
            cnv1->updateKernels(lr);

            //.print();
            // d2.print(); // GOOD

            // l2->dw = l2->getDelta(d2, a);
            // l2->db = d2;

            l0->UpdateWeights(lr);
            //l2->UpdateWeights(lr);

            accuracy = c.argmax() == y[i].argmax() ? accuracy + 1 : accuracy;

            error += CrossEntropy(y[i], c);
            //std::printf("\nMemory USED %d\n", MemoryArena.Used);
            MemoryArena.Used = usedMem;
        }

        MemoryArena.Used = usedMem;
        std::printf("\nEpoch(%d/%d) Train Error: %f, Train Acc: %f, Valid Error: %f, Valid Acc: %f", 
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
            M a = Sigmoid(l0->forward(xtest2[i]));
            M c = Softmax(l2->forward(a));    
            
            v_accuracy = c.argmax() == ytest[i].argmax() ? v_accuracy + 1 : v_accuracy;

            v_error += CrossEntropy(ytest[i], c);

            MemoryArena.Used = usedMem;
        }
        for(u32 i=0;i<M_EXAMPLES;++i){

            M a = Sigmoid(l0->forward(input2[i]));
            M c = Softmax(l2->forward(a));    
            
            M d2 = M::MatMul(CrossEntropyPrime(y[i], c), SoftmaxPrime(c));
            M d0 = l2->backward(d2) * SigmoidPrime(a);

            l0->dw = l0->getDelta(d0, input2[i]);
            l0->db = d0;

            l2->dw = l2->getDelta(d2, a);
            l2->db = d2;

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
