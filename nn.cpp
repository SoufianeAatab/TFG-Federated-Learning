#include <iostream>
#include <assert.h>
#include <chrono>
#include <random>
#include <stdint.h>

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
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-0.5,0.5);

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

    static M ones(u32 rows, u32 cols){
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        memset(data,1, rows*cols*sizeof(f32));
        M o(data, rows, cols);
        return o;
    }

    static M zeros(u32 rows, u32 cols){
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        memset(data, 0, rows*cols*sizeof(f32));
        return M(data, rows, cols);
    }

    static M rand(u32 rows, u32 cols){
        f32* data = (f32*) PushSize(&MemoryArena, rows*cols*sizeof(f32));
        M o(data, rows,cols);
        for(u32 i=0;i<rows*cols;++i){
            o.data[i] = distribution(generator);
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

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define PORT 65432
f32* get_data(u32* size){
    int server_fd;
    int new_socket;
    struct sockaddr_in address;
    int addrlen;
    int opt = 1; 
    addrlen = sizeof(address); 
    std::cout << "Waiting for incoming connection...\n";
    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    }

    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR , &opt, sizeof(opt)))
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( PORT ); 

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) 
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
                        (socklen_t*)&addrlen)) < 0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    }
    std::cout << "Connected by " << inet_ntoa(address.sin_addr) << "\n";
    // receive 4 bytes for length indicator
    int bytes_length_count = 0;
    int bytes_length_total = 0;
    
    uint32_t length_descriptor = 0;
    char* len_buffer = reinterpret_cast<char*>(&length_descriptor);
    
    while (bytes_length_total < 4)
    {
        bytes_length_count = recv(new_socket,
                                    &len_buffer[bytes_length_total],
                                    sizeof (uint32_t) - bytes_length_total,
                                    0);
        
        if (bytes_length_count == -1)
        {
            perror("recv");
        }
        else if (bytes_length_count == 0)
        {
            std::cout << "Unexpected end of transmission." << " 1.Received: " << bytes_length_total << std::endl;
            close(server_fd);
            exit(EXIT_SUCCESS);
        }

        bytes_length_total += bytes_length_count;
    }

    
    // receive payload
    int bytes_payload_count = 0;
    int bytes_payload_total = 0;
    
    size_t data_size = length_descriptor*sizeof(f32);
    f32* data = PushSize(&MemoryArena, data_size);
    char* buffer = reinterpret_cast<char*>(&data[0]);
    std::cout << "Want to receive " << data_size << std::endl;
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
            std::cout << "Unexpected end of transmission." << " Received: " << bytes_length_total << std::endl;
            close(server_fd);
            exit(EXIT_SUCCESS);
        }
        bytes_payload_total += bytes_payload_count;
    }
    
    *size = length_descriptor;
    close(server_fd);
    return data; 
}

int main(){
    // Reserve memory arena which we'll use
    InitMemory(1024*1024*1024);

    u32 XSize, YSize;;
    f32* xs = get_data(&XSize);
    f32* ys = get_data(&YSize);
    M x[600] = {};
    M y[600] = {};
    
    for(u32 i=0;i<600;++i){
        x[i] = {xs + (i), 1, 1};
        y[i] = {ys + (i), 1, 1};
    }


    // for(u32 i=60000;i<70000;++i){
    //     xv[i-60000] = {xs + (i*784), 1, 784};
    //     yv[i-60000] = {ys + (i*10),  1, 10};
    // }

    Layer l1(1,8);  
    Layer l2(8,8);    
    Layer l3(8,1);    
   
    u32 MemoryUsed = MemoryArena.Used;
    std::cout << "Memory used for NN initialization: " << MemoryArena.Used << "\n";
    u32 epochs = 500;
    f32 lr = 0.1;
    for(u32 i=0;i<epochs;++i){
        f32 error = 0.0f;
        for(u32 j=0;j<600;++j){
            M a = Sigmoid(l1.forward(x[j]));
            M h = Sigmoid(l2.forward(a));
            M o = l3.forward(h);

            M e3 = Loss(y[j], o);
            M d3 = e3;

            M e2 = l3.backward(d3);
            M d2 = e2 * SigmoidPrime(h);

            M e1 = l2.backward(d2);
            M d1 = e1 * SigmoidPrime(a);

            l3.UpdateWeights(d3, h, lr);
            l2.UpdateWeights(d2, a, lr);
            l1.UpdateWeights(d1, x[j], lr);

            error += Mse(y[j], o);
        }
        if(i%10==0){
            std::cout << i << "/" << epochs << "Loss: " << error/600.0f << std::endl; 
        }

        // validation set test
        // f32 accuracy = 0.0f;
        // for(u32 k=0;k<10000;++k){
        //     M a = Sigmoid(l1.forward(xv[k]));
        //     M h = Sigmoid(l2.forward(a));
        //     M o = Softmax(l3.forward(h));
        //     u32 omax = o.argmax();
        //     u32 ymax = yv[k].argmax();
        //     accuracy += omax == ymax ? 1.0f : 0.0f;
        //     MemoryArena.Used = MemoryUsed;
        // }
        // std::cout << "Accuracy: " << accuracy/10000.0f << "\n";

        // Optimization algorithm
        // f32 error = 0.0f;
        // for(u32 j=0;j<60000;++j){
        //     // Forward pass
        //     M a = Sigmoid(l1.forward(x[j]));
        //     M h = Sigmoid(l2.forward(a));
        //     M o = Softmax(l3.forward(h));

        //     // Backward pass
        //     M e3 = CrossEntropyPrime(y[j], o);
        //     M d3 = M::MatMul(e3, SoftmaxPrime(o));

        //     M e2 = l3.backward(d3);
        //     M d2 = e2 * SigmoidPrime(h);

        //     M e1 = l2.backward(d2);
        //     M d1 = e1 * SigmoidPrime(a);

        //     // Update pass
        //     l3.UpdateWeights(d3, h, lr);
        //     l2.UpdateWeights(d2, a, lr);
        //     l1.UpdateWeights(d1, x[j], lr);

        //     error += CrossEntropy(y[j], o);
        //     //if(i==0 && j==0) std::cout << "Memory used for training: " << MemoryArena.Used << "\n";
        //     MemoryArena.Used = MemoryUsed;
        // }
        // if(i!= 0 && i%50 == 0) lr *= 0.1;
    
        // std::cout << "LOSS: " <<  error / (60000.0f) << "[" << i << "/" << epochs <<"]"<< std::endl;
        // std::cout << "------------------------------\n";
    }
    // Free Memory

    free(MemoryArena.Base);
}