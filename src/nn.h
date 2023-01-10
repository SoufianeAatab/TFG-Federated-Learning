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

struct subscript{
    u32 row;
    u32 col;
};

struct memory_arena{
    size_t Used;
    size_t Size;
    u8 *Base;
};
static memory_arena MemoryArena = {};

inline f32 *PushSize(memory_arena *Arena, size_t SizeToReserve)
{   
    // std::cout << "want to reserve"<< SizeToReserve << "\n";
    assert(Arena->Used + SizeToReserve <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += SizeToReserve;
    return (f32*)Result;
}

void InitMemory(u32 Size){
    MemoryArena.Base = (u8 *)malloc(Size);
    MemoryArena.Size = Size;
    MemoryArena.Used = 0;
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
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        memset(data,1, rows*cols);
        M o(data, rows, cols);
        return o;
    }

    static M zeros(u32 rows, u32 cols){
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        memset(data, 0, rows*cols);
        return M(data, rows, cols);
    }

    static M rand(u32 rows, u32 cols){
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        M o(data, rows,cols);
        for(u32 i=0;i<rows*cols;++i){
            o.data[i] = ((random(0,10000) - 5000.0f) / 10000.0f);
        }
        return o;
    }
  
    void print(){
        std::cout << "SHAPE: (" << rows <<"," << stride << ")" << std::endl;
        for(u32 i=0;i<size;++i){
            for(u32 j=0;j<stride;++j){
                std::cout << data[i*stride+j] << " ";
            }
            std::cout << "\n";
        }
    }
    
    void transpose(){
        if(stride != 1){
            u32 aux = stride;
            stride = size;
            size = aux;
        }
    }
    
    f32& operator[](subscript idx){
        return data[idx.row * stride + idx.col];
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
            this->data[i]+=b.data[i];
        }
    }

    void sub(M b){
        assert(size == b.size && stride==b.stride);
        for(u32 i=0;i<b.size*b.stride;++i){
            this->data[i]-=b.data[i];
        }
    }

    float sum(){
        f32 s = 0.0f;
        for(u32 i=0;i<size*stride;++i){
            s += data[i];
        }
        return s;
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

    M square(){
        M out = M::zeros(size, stride);
        for(u32 i=0;i<size*stride;++i){
            out.data[i] = data[i] * data[i];
        }
        return out;
    }

    static M MatMul(M A, M B){
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

    static void MatMul_(M A, M B, M& Out){
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

    static Layer* Create(u32 input, u32 output){
        Layer* l = (Layer*)PushSize(&MemoryArena, sizeof(Layer));
        l->w = M::rand(input, output);
        l->b = M::zeros(1, output);
        return l;
    }

    Layer(u32 input, u32 output): w(M::rand(input, output)), b(M::zeros(1, output)){}

    M forward(M x){
        assert(x.cols == w.rows);
        M Out = M::zeros(b.rows, b.cols);
        for(u32 i=0;i<w.cols;++i)
        {
            f32 accum = 0;
            for(u32 j=0;j<x.cols;++j)
            {
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
            for (u32 l=0;l<w.cols;++l)
            {
                accum += grad.data[l] * w.data[l*w.rows+k];

            }
            out.data[k] = accum;
        }
        return out;
    }

    void UpdateWeights(M grads, M a, f32 lr){
        for (u32 i=0;i<w.rows;++i){
            for (u32 j=0;j<w.cols;++j){
                w.data[i * w.cols + j] -= lr * grads.data[j] * a.data[i];
            }
            b.data[i] -= lr * grads.data[i];
        }
    }
};


f32 CrossEntropy(M y, M y_hat) {
  f32 loss = 0;
  for (u32 i = 0; i < y.cols; i++) {
    loss += y[i] * log(y_hat[i] + 1e-9);
  }
  return -loss;
}

M CrossEntropyPrime(M y, M y_hat){
    M out = M::zeros(y.rows, y.cols);
    for (u32 i = 0; i < y.cols; i++) {
        out.data[i] = -y[i] / (y_hat[i] + 1e-15);
    }
    return out;
    // return -(y / y_hat);
}

M Loss(M y, M y_hat){
    return y_hat - y;
}

f32 Mse(M y, M y_hat)
{
    return (y-y_hat).square().sum();
}

M Softmax(M X){
    f32 sum = 0.0f;
    f32 max = X.data[X.argmax()];
    M out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.cols;++i){
        sum += exp(X.data[i] - max);
    }
    for(u32 i=0;i<X.cols;++i){
        out.data[i] = exp(X.data[i] - max) / sum;
    }
    return out;
}

M SoftmaxPrime(M X){
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

f32 Sigmoid_(f32 x){
    return 1.0f / (1.0f + exp(-x));
}

M Sigmoid(M X){
    u32 Cols = X.cols;
    u32 Rows = X.rows;
    for(u32 i=0;i<Rows * Cols;++i){
        X.data[i] = Sigmoid_(X.data[i]);
    }
    return X;
}

f32 D_Sigmoid(f32 X){
    return X * (1.0f - X);
}

M SigmoidPrime(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = D_Sigmoid(X.data[i]);
    }
    return Out;
}

M Relu(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
    }
    return Out;
}

M ReluPrime(M X){
    M Out = M::zeros(X.rows, X.cols);
    for(u32 i=0;i<X.rows * X.cols;++i){
        Out.data[i] = X.data[i] > 0.0f ? 1.0f : 0.0f;
    }
    return Out;
}
#endif