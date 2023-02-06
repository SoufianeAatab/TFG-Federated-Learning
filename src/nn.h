#ifndef NN_H
#define NN_H
#include <iostream>
#include <assert.h>
#include <chrono>
#include <stdint.h>
#include <Arduino.h>

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

struct MemoryArena{
    size_t used;
    size_t size;
    u8 *base;
};

static MemoryArena memoryArena = {};

inline f32 *pushSize(MemoryArena *arena, size_t sizeToReserve)
{   
    assert(arena->used + sizeToReserve <= arena->size);
    void *result = arena->base + arena->used;
    arena->used += sizeToReserve;
    return (f32*)result;
}

void initMemory(u32 size){
    memoryArena.base = (u8 *)malloc(size);
    memoryArena.size = size;
    memoryArena.used = 0;
}

struct M{
    f32* data;  
    union{
        struct{
            u32 size;
            u32 stride;
        };
        struct {
            u32 rows;
            u32 cols;
        };
    };

    //M (const M&) = delete;
    //M& operator= (const M&) = delete;
    M() = default;
    M(f32* data, u32 rows, u32 cols):data(data), size(rows), stride(cols){}

    static void memset(f32* data, f32 val, u32 size){
        for(u32 i=0;i<size;++i) data[i] = val;
    }

    static M ones(u32 rows, u32 cols){
        f32* data = (f32*) pushSize(&memoryArena, rows*cols*sizeof(f32));
        memset(data,1, rows*cols);
        M o(data, rows, cols);
        return o;
    }

    static M zeros(u32 rows, u32 cols){
        f32* data = (f32*) pushSize(&memoryArena, rows*cols*sizeof(f32));
        memset(data, 0, rows*cols);
        return M(data, rows, cols);
    }

    static M rand(u32 rows, u32 cols){
        f32* data = (f32*) pushSize(&memoryArena, rows*cols*sizeof(f32));
        M o(data, rows,cols);
        for(u32 i=0;i<rows*cols;++i){
            o.data[i] = ((random(0,10000) - 5000.0f) / 10000.0f);
        }
        return o;
    }
  
    void print(){
        Serial.print("SHAPE: (");
        Serial.print(rows);
        Serial.print(",");
        Serial.print(stride);

        Serial.println(")");
        for(u32 i=0;i<size;++i){
            for(u32 j=0;j<stride;++j){
                Serial.print(data[i*stride+j]);
                Serial.print(" ");
            }
            Serial.println();
        }
    }
    
    void transpose(){
        if(stride != 1){
            u32 aux = stride;
            stride = size;
            size = aux;
        }
    }

    f32& operator[](u32 idx){
        return data[idx];
    }

    M operator-(M B){
        M out = M::zeros(rows, cols);
        for(u32 i=0;i<rows*cols;++i){
            out.data[i] = data[i] - B.data[i];
        }
        return out;
    }

    M operator-(){
        M out = M::zeros(rows, cols);
        for(u32 i=0;i<rows*cols;++i){
            out.data[i] = -data[i];
        }
        return out;
    }

    M operator/(M B){
        M out = M::zeros(rows, cols);
        for(u32 i=0;i<rows*cols;++i){
            out.data[i] = data[i] * (1.0f/B.data[i]);
        }
        return out;
    }

    M operator*(M B){
        M out = M::zeros(rows, cols);
        for(u32 i=0;i<rows*cols;++i){
            out.data[i] = data[i] * B.data[i];
        }
        return out;
    }

    void add(M b){
        assert(size == b.size && stride==b.stride);
        for(u32 i=0;i<b.size*b.stride;++i){
            data[i]+=b.data[i];
        }
    }

    void sub(M b){
        assert(size == b.size && stride==b.stride);
        for(u32 i=0;i<b.size*b.stride;++i){
            data[i]-=b.data[i];
        }
    }

    float sum(){
        f32 result = 0.0f;
        for(u32 i=0;i<size*stride;++i){
            result += data[i];
        }
        return result;
    }

    u32 argmax(){
        u32 imax = 0;
        for(u32 i=0;i<rows*cols;++i){
            if(data[i] > data[imax]){
                imax = i;
            }
        }
        return imax;
    }

    u32 argmin(){
        u32 imin = 0;
        for(u32 i=0;i<rows*cols;++i){
            if(data[i] < data[imin]){
                imin = i;
            }
        }
        return imin;
    }

    M square(){
        M out = M::zeros(size, stride);
        for(u32 i=0;i<size*stride;++i){
            out.data[i] = data[i] * data[i];
        }
        return out;
    }

    static M matMul(M A, M B){
        assert(A.stride == B.size);
        M Out = M::zeros(A.size, B.stride);
        for (u32 i = 0; i < A.size; i++) {
            for (u32 j = 0; j < B.stride; j++) {
                Out.data[i* B.stride + j] = 0;
                for (u32 k = 0; k < A.stride; k++) {
                    Out.data[i* B.stride + j] += A.data[i * A.stride + k] * B.data[k*B.stride + j];  
                }
            }
        }
        return Out;
    }

    static void matMul_(M A, M B, M& Out){
        for (u32 i = 0; i < A.size; i++) {
            for (u32 j = 0; j < B.stride; j++) {
                Out.data[i* B.stride + j] = 0;
                for (u32 k = 0; k < A.stride; k++) {
                    Out.data[i* B.stride + j] += A.data[i * A.stride + k] * B.data[k*B.stride + j];  
                }
            }
        }
    }
};

struct Layer{
    M w;
    M b;

    static Layer* create(u32 input, u32 output){
        Layer* l = (Layer*)pushSize(&memoryArena, sizeof(Layer));
        l->w = M::rand(input, output);
        l->b = M::zeros(1, output);
        return l;
    }

    //Layer(u32 input, u32 output): w(M::rand(input, output)), b(M::zeros(1, output)){}

    M forward(M x){
        assert(x.cols == w.rows);
        M Out = M::zeros(b.rows, b.cols);
        for(u32 i=0;i<w.cols;++i) {
            f32 accum = 0;
            for(u32 j=0;j<x.cols;++j) {
                accum += x.data[j] * w.data[j*w.cols + i];
            }
            Out.data[i] = accum + b.data[i];
        }
        return Out;
    }

    M backward(M grad){
        M out = M::zeros(1,w.rows);
        for(u32 k=0;k<w.rows;++k) {
            f32 accum = 0;
            for (u32 l=0;l<w.cols;++l) {
                accum += grad.data[l] * w.data[l*w.rows+k];
            }
            out.data[k] = accum;
        }
        return out;
    }

    void updateWeights(M grads, M a, f32 lr){
        for (u32 i=0;i<w.rows;++i){
            for (u32 j=0;j<w.cols;++j){
                w.data[i * w.cols + j] -= lr * grads.data[j] * a.data[i];
            }
            b.data[i] -= lr * grads.data[i];
        }
    }
};

// #if defined ( __GNUC__ )
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wstrict-aliasing"
// #endif
//     /**
//      * > 50% faster then the math.h log() function
//      * in return for a small loss in accuracy (0.00001 average diff with log())
//      * From: https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c/39822314#39822314
//      * Licensed under the CC BY-SA 3.0
//      * @param a Input number
//      * @returns Natural log value of a
//      */
//     __attribute__((always_inline)) static inline float log(float a)
//     {
//         float m, r, s, t, i, f;
//         int32_t e, g;

//         g = (int32_t) * ((int32_t *)&a);
//         e = (g - 0x3f2aaaab) & 0xff800000;
//         g = g - e;
//         m = (float) * ((float *)&g);
//         i = (float)e * 1.19209290e-7f; // 0x1.0p-23
//         /* m in [2/3, 4/3] */
//         f = m - 1.0f;
//         s = f * f;
//         /* Compute log1p(f) for f in [-1/3, 1/3] */
//         r = fmaf(0.230836749f, f, -0.279208571f); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
//         t = fmaf(0.331826031f, f, -0.498910338f); // 0x1.53ca34p-2, -0x1.fee25ap-2
//         r = fmaf(r, s, t);
//         r = fmaf(r, s, f);
//         r = fmaf(i, 0.693147182f, r); // 0x1.62e430p-1 // log(2)

//         return r;
//     }
// #if defined ( __GNUC__ )
// #pragma GCC diagnostic pop
// #endif

f32 crossEntropy(M y, M y_hat) {
  f32 loss = 0;
  for (u32 i = 0; i < y.cols; i++) {
    // adding epsilon because log of 0 returns NaN
    loss += y[i] * log(y_hat[i] + 1e-9);
  }
  return -loss;
}

M crossEntropyPrime(M y, M y_hat){
    M out = M::zeros(y.rows, y.cols);
    for (u32 i = 0; i < y.cols; i++) {
        // adding epsilon(1e-15) preventing division by zero
        out.data[i] = -y[i] / (y_hat[i] + 1e-15);
    }
    return out;
}

M msePrime(M y, M y_hat){
    return y_hat - y;
}

f32 mse(M y, M y_hat)
{
    return (y-y_hat).square().sum();
}

M softmax(M X){
    f32 max = X.data[X.argmax()];
    M out = M::zeros(X.rows, X.cols);
    
    f32 sum = 0.0f;
    // subtracting max, preventing +inf error
    for(u32 i=0;i<X.cols;++i){
        sum += exp(X.data[i] - max);
    }

    for(u32 i=0;i<X.cols;++i){
        out.data[i] = exp(X.data[i] - max) / sum;
    }
    return out;
}

M softmaxPrime(M X){
    M Out = M::zeros(X.cols, X.cols);
    for(u32 i=0;i<X.cols;++i){
        for(u32 j=0;j<X.cols;++j){
            if(i==j){
                Out.data[i * X.cols + j] = X.data[i] * (1.0f -X.data[i]);
            } else{
                Out.data[i * X.cols + j] = -X.data[i] * X.data[j];
            }
        }    
    }
    return Out;
}

M sigmoid(M X){
    u32 Cols = X.cols;
    u32 Rows = X.rows;
    for(u32 i=0;i<Rows * Cols;++i){
        X.data[i] = 1.0f / (1.0f + exp(-X.data[i]));
    }
    return X;
}

M sigmoidPrime(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = X.data[i] * (1.0f - X.data[i]);
    }
    return Out;
}

M relu(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M reluPrime(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = X.data[i] > 0.0f ? 1.0f : 0.0f;
    }
    return Out;
}
#endif