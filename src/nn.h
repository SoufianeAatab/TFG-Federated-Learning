#ifndef N_HH
#define N_HH
#include <iostream>
#include <assert.h>
#include <chrono>
#include <stdint.h>

#if ONCOMPUTER
#include <random>
#include <fstream>

std::random_device rd;
std::mt19937 generator(0);
std::uniform_real_distribution<double> distribution;

#else
#include <Arduino.h>
#endif
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

static f32 randLowVal = 0;
static f32 randHighVal = 0;
static f32 scale = 0.0f;




#if ONCOMPUTER
f32 uniform_rand(){
    static std::mt19937 gen(1);
    //std::printf("between %f, %f\n", randLowVal, randHighVal);
    std::uniform_real_distribution<f32> dst(randLowVal, randHighVal);
    return dst(gen);
}
#else
f32 uniform_rand(){
    float num = (float)random(1001) / 1000.0 - randLowVal;
    return num;
}
#endif

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
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            o.data[i] = uniform_rand();
            #endif
        }
        return o;
    }

    void shape()
    {
        std::printf("Shape: (%lu, %lu)\n", rows, cols);
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

    M operator+(M B)
    {
        M out = M::zeros(rows, cols);
        for (u32 i = 0; i < rows * cols; ++i)
        {
            out.data[i] = data[i] + B.data[i];
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

    f32 std(){
        f32 sum = 0.0f;
        u32 n = rows * cols;
        for (u32 i = 0; i < n; ++i)
        {
            sum += data[i];
        }

        f32 mean = sum / n;
        f32 sd = 0.0f;
        for(u32 i = 0; i < n; i++) {
            sd += pow(data[i] - mean, 2);
        }

        return sqrt(sd / (f32)n);
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

    #if ONCOMPUTER
    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->rows;
        outfile << "\t";
        outfile << this->cols;
        outfile << "\t";
        for(u32 i=0;i<this->rows*this->cols;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->rows;
        infile >> this->cols;
        for(u32 i=0;i<this->rows * this->cols;++i){
            infile >> this->data[i];
        }
        infile.close();
    }
    #endif
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
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            o.data[i] = uniform_rand();
            #endif       
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
        assert(B.d1 == d1 && B.d2 == d2 && B.d3 == d3);
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    M3 operator*(f32 B)
    {
        M3 out = M3::zeros(d1,d2,d3);
        for (u32 i = 0; i < d1 * d2 * d3; ++i){
            out.data[i] = data[i] * B;
        }
        return out;
    }

    M flatten(){
        return M(data, 1, d1*d2*d3);
    }

    void shape(){
        std::printf("SHAPE: %lu, %lu, %lu\n",d1,d2,d3 );
    }

    void print(){
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

    #if ONCOMPUTER

    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->d1;
        outfile << "\t";
        outfile << this->d2;
        outfile << "\t";
        outfile << this->d3;
        outfile << "\t";
        for(u32 i=0;i<this->d1*this->d2*this->d3;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->d1;
        infile >> this->d2;
        infile >> this->d3;
        for(u32 i=0;i<this->d1 * this->d2 * this->d3;++i){
            infile >> this->data[i];
        }
        infile.close();
    }
    #endif

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
            #if ONCOMPUTER
            o.data[i] = uniform_rand();
            #else
            //o.data[i] = uniform_rand();
            #endif        
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

    M4 operator+(M4 B)
    {
        M4 out = M4::zeros(d1, d2, d3, d4);
        for (u32 i = 0; i < d1 * d2 * d3 * d4; ++i)
        {
            out.data[i] = data[i] + B.data[i];
        }
        return out;
    }

    f32 std(){
        f32 sum = 0.0f;
        u32 n = d1 * d2 * d3 * d4;
        for (u32 i = 0; i < d1 * d2 * d3 * d4; ++i)
        {
            sum += data[i];
        }

        f32 mean = sum / n;
        f32 sd = 0.0f;
        for(u32 i = 0; i < d1 * d2 * d3 * d4; i++) {
            sd += pow(data[i] - mean, 2);
        }

        return sqrt(sd / (f32)n);
    }

    void print(){
        std::printf("SHAPE: %lu, %lu, %lu, %lu\n",d1,d2,d3,d4 );

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
    #if ONCOMPUTER

    void store(const char* filename){
        std::ofstream outfile(filename);

        outfile << this->d1;
        outfile << "\t";
        outfile << this->d2;
        outfile << "\t";
        outfile << this->d3;
        outfile << "\t";
        outfile << this->d4;
        outfile << "\t";
        for(u32 i=0;i<this->d1*this->d2*this->d3*this->d4;++i){
            outfile << this->data[i];
            outfile << " ";
        }
        outfile.close();
    }

    void load(const char* filename){
        std::ifstream infile(filename);
        infile >> this->d1;
        infile >> this->d2;
        infile >> this->d3;
        infile >> this->d4;
        for(u32 i=0;i<this->d1 * this->d2 * this->d3*this->d4;++i){
            infile >> this->data[i];
        }
        infile.close();
    }
    #endif
};
#define MT_N 624
#define MT_M 397

typedef struct {
    unsigned int mt[MT_N];
    int index;
} mt_state;
mt_state mystate;
void mt_seed(mt_state *state, unsigned int seed) {
    state->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        state->mt[i] = 1812433253 * (state->mt[i-1] ^ (state->mt[i-1] >> 30)) + i;
    }
    state->index = MT_N;
}

void mt_twist(mt_state *state) {
    static const unsigned int MAG01[2] = {0, 0x9908b0df};
    unsigned int y;

    for (int i = 0; i < MT_N; i++) {
        y = (state->mt[i] & 0x80000000) | (state->mt[(i+1) % MT_N] & 0x7fffffff);
        state->mt[i] = state->mt[(i + MT_M) % MT_N] ^ (y >> 1) ^ MAG01[y & 0x1];
    }
    state->index = 0;
}

unsigned int mt_rand(mt_state *state) {
    if (state->index >= MT_N) {
        mt_twist(state);
    }
    unsigned int y = state->mt[state->index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
}

double mt_rand_double(mt_state *state) {
    return (double)mt_rand(state) / (double)0xffffffff;
}

double mt_rand_xavier(mt_state *state, f32 low, f32 high) {
    return low + (mt_rand_double(state) * (high - low));
}

void XavierInitialization(M4 data, f32 fan_in, f32 fan_out, f32 min_val, f32 max_val){
    // Calculate the scaling factor based on the fan-in and fan-out
    //float scale = sqrt(6.0 / (fan_in + fan_out));
    
    // Initialize the weights with random values from a uniform distribution
    // std::random_device rd;
    // std::mt19937 gen(0);
    // std::uniform_real_distribution<float> dist(min_val, max_val);
    f32 scale = (sqrt(6.0) / sqrt(fan_out + fan_in));

    for(u32 i=0;i<data.d1 * data.d2 * data.d3 * data.d4;++i){
        data.data[i] = mt_rand_xavier(&mystate, -scale, scale);
    }
}

void XavierInitialization(M data, f32 fan_in, f32 fan_out, f32 min_val, f32 max_val){
    // Calculate the scaling factor based on the fan-in and fan-out
    //float scale = sqrtf(6.0 / (fan_in + fan_out));
    
    // Initialize the weights with random values from a uniform distribution
    // std::random_device rd;
    // std::mt19937 gen(0);
    // std::uniform_real_distribution<float> dist(min_val, max_val);
    f32 scale = (sqrt(6.0) / sqrt(fan_out + fan_in));
    for(u32 i=0;i<data.rows * data.cols;++i){
        data.data[i] = mt_rand_xavier(&mystate, -scale, scale);
    }
}


// This function computes the Cross Entropy loss between two sets of outputs, y and y_hat.
f32 CrossEntropy(M y, M y_hat)
{
    // Initialize the loss to zero.
    f32 loss = 0;
    // Loop through each output in the target y and predicted y_hat.
    for (u32 i = 0; i < y.cols; i++)
    {
        // The 1e-9 is added to avoid log(0) which would result in a NaN (Not a Number) value.
        loss += y[i] * log(y_hat[i] + 1e-9);
    }
    // Return the negative of the loss.
    return -loss;
}

// This function computes the derivative of the Cross Entropy loss with respect to the predicted outputs, y_hat.
M CrossEntropyPrime(M y, M y_hat)
{  
    // Initialize an empty matrix with the same shape as the inputs, y and y_hat.
    M out = M::zeros(y.rows, y.cols);
    for (u32 i = 0; i < y.cols; i++)
    {
        // The 1e-15 is added to avoid division by zero.
        out.data[i] = -y[i] / (y_hat[i] + 1e-15);
    }
    return out;
    // return -(y / y_hat);
}

f32 BinaryCrossEntropyLoss(M y, M y_hat)
{
    f32 loss = 0;
    
    for (u32 i = 0; i < y.cols; i++)
    {
        // Add a small value to avoid taking the log of zero.
        const f32 eps = 1e-15;
        loss += -(y[i] * log(y_hat[i] + eps) + (1 - y[i]) * log(1 - y_hat[i] + eps));
    }
    
    return loss / y.cols;
}

M BinaryCrossEntropyPrime(M y, M y_hat)
{
    // Initialize an empty matrix with the same shape as the inputs, y and y_hat.
    M out = M::zeros(y.rows, y.cols);
    
    for (u32 i = 0; i < y.cols; i++)
    {
        // Add a small value to avoid division by zero.
        const double eps = 1e-15;
        out.data[i] = (y_hat[i] - y[i]) / (y_hat[i] * (1 - y_hat[i]) + eps);
    }
    
    return out;
}

// Computes the derivative of the mean squared error (MSE) loss with respect to 'y_hat'
inline M MsePrime(M y, M y_hat)
{
    return y_hat - y;
}

// Computes the mean squared error (MSE) loss between two matrices 'y' and 'y_hat'
inline f32 Mse(M y, M y_hat)
{
    return (y - y_hat).square().mean();
}

// Softmax function calculates the probability of each output for a given input matrix
M Softmax(M X)
{
    f32 sum = 0.0f;
    f32 max = X.data[X.argmax()];
    M out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.cols; ++i)
    {
        sum += exp(X.data[i] - max);
    }
    for (u32 i = 0; i < X.cols; ++i)
    {
        // subtract max to avoid NaN/+inf errors
        out.data[i] = exp(X.data[i] - max) / sum;
    }
    return out;
}

// SoftmaxPrime function calculates the derivative of the softmax function
M SoftmaxPrime(M X)
{
    M Out = M::zeros(X.cols, X.cols);
    // calculate the derivative of the softmax function
    for (u32 i = 0; i < X.cols; ++i)
    {
        for (u32 j = 0; j < X.cols; ++j)
        {
            if (i == j)
            {
                Out.data[i * X.cols + j] = X.data[i] * (1.0f - X.data[i]);
            }
            else
            {
                Out.data[i * X.cols + j] = -X.data[i] * X.data[j];
            }
        }
    }
    return Out;
}

M Sigmoid(M X)
{
    u32 Cols = X.cols;
    u32 Rows = X.rows;
    for (u32 i = 0; i < Rows * Cols; ++i)
    {
        X.data[i] = 1.0f / (1.0f + exp(-X.data[i]));
    }
    return X;
}

M SigmoidPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] * (1.0f - X.data[i]);
    }
    return Out;
}

M3 Sigmoid(M3 X)
{
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        X.data[i] = 1.0f / (1.0f + exp(-X.data[i]));
    }
    return X;
}

M3 SigmoidPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] * (1.0f - X.data[i]);
    }
    return Out;
}

M Relu(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M ReluPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? 1.0 : 0.0f;
    }
    return Out;
}

M3 Relu(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M3 ReluPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1 * X.d2 * X.d3; ++i)
    {
        Out.data[i] = X.data[i] > 0.0f ? 1.0 : 0.0f;
    }
    return Out;
}

M Tanh(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = tanh(X.data[i]);
    }
    return Out;
}

M TanhPrime(M X)
{
    M Out = M::zeros(X.rows, X.cols);
    for (u32 i = 0; i < X.rows * X.cols; ++i)
    {
        Out.data[i] = (1 - X.data[i] * X.data[i]);
    }
    return Out;
}

M3 Tanh(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i <  X.d1*X.d2*X.d3; ++i)
    {
        Out.data[i] = tanh(X.data[i]);
    }
    return Out;
}

M3 TanhPrime(M3 X)
{
    M3 Out = M3::zeros(X.d1, X.d2, X.d3);
    for (u32 i = 0; i < X.d1*X.d2*X.d3; ++i)
    {
        Out.data[i] = (1 - X.data[i] * X.data[i]);
    }
    return Out;
}


void setRandomUniform(double low, double high)
{
    #if ONCOMPUTER
        distribution = std::uniform_real_distribution<double>(low, high);
    #else
        randLowVal = low;
        randHighVal = high;
    #endif
}

// Set Glorot uniform distribution for weights random intialization
void setGlorotUniform(u32 in, u32 out)
{
    double scale = sqrt(6.0f / ((f32)in + (f32)out));
    #if ONCOMPUTER
    randLowVal = -scale;
    randHighVal = scale;  
    #else
    randLowVal = -scale;
    randHighVal = scale;
    #endif
    
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
        //setRandomUniform(-0.5, 0.5);
#endif
        // Initialize the weight and bias matrices with random values
        l->w = M::rand(input_size, output_size);
        XavierInitialization(l->w, input_size, output_size, -1.0f, 1.0f);
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
            b.data[i] -= lr * this->vdb.data[i];
        }
    }

    void UpdateWeights(M grads, M a, f32 lr)
    {
        // MOMENTUM
        vdw = vdw * 0.9f + dw * (1.0f-0.9f);  
        vdb = vdb * 0.9f + db * (1.0f-0.9f);  

        for (u32 i = 0; i < w.rows; ++i)
        {
            u32 index = i * w.cols;
            for (u32 j = 0; j < w.cols; ++j)
            {
                //w.data[i * w.cols + j] -= lr * this->dw[i * this->dw.cols + j];
                dw.data[index + j] = grads.data[j] * a.data[i];
                // update momentum vel 
                vdw.data[index + j] = vdw.data[index + j] * 0.9f + dw.data[index + j] * (1.0f-0.9f);
                // Update weights
                w.data[index + j] -= lr * this->vdw.data[index + j];
            }
            db.data[i] = grads.data[i];
            // update momentum vel
            vdb.data[i] = vdb.data[i] * 0.9f + db.data[i] * (1.0f-0.9f);  
            // Update bias
            b.data[i] -= lr * this->vdb.data[i];
        }
    }
};

#endif

