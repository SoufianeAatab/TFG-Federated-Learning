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

struct subscript
{
    u32 row;
    u32 col;
};

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

inline f32 *PushSize(memory_arena *Arena, size_t SizeToReserve)
{
    // std::cout << "want to reserve"<< SizeToReserve << "\n";
    assert(Arena->Used + SizeToReserve <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += SizeToReserve;
    return (f32 *)Result;
}

void InitMemory(u32 Size)
{
    MemoryArena.Base = (u8 *)malloc(Size);
    MemoryArena.Size = Size;
    MemoryArena.Used = 0;
}

struct M
{
    f32 *data;
    union
    {
        struct
        {
            u32 size;
            u32 stride;
        };
        struct
        {
            u32 rows;
            u32 cols;
        };
    };

    // M (const M&) = delete;
    // M& operator= (const M&) = delete;
    M() = default;
    M(f32 *data, u32 rows, u32 cols) : data(data), size(rows), stride(cols) {}

    static M ones(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        memset(data, 1, rows * cols * sizeof(f32));
        M o(data, rows, cols);
        return o;
    }

    static M zeros(u32 rows, u32 cols)
    {
        f32 *data = (f32 *)PushSize(&MemoryArena, rows * cols * sizeof(f32));
        memset(data, 0, rows * cols * sizeof(f32));
        return M(data, rows, cols);
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
        for (u32 i = 0; i < size; ++i)
        {
            for (u32 j = 0; j < stride; ++j)
            {
                std::printf("%f ", data[i * stride + j]);
            }
            std::printf("\n");
        }
    }

    void transpose()
    {
        if (stride != 1)
        {
            u32 aux = stride;
            stride = size;
            size = aux;
        }
    }

    f32 &operator[](subscript idx)
    {
        return data[idx.row * stride + idx.col];
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
        assert(size == b.size && stride == b.stride);
        for (u32 i = 0; i < b.size * b.stride; ++i)
        {
            this->data[i] += b.data[i];
        }
    }

    void sub(M b)
    {
        assert(size == b.size && stride == b.stride);
        for (u32 i = 0; i < b.size * b.stride; ++i)
        {
            this->data[i] -= b.data[i];
        }
    }

    float sum()
    {
        f32 s = 0.0f;
        for (u32 i = 0; i < size * stride; ++i)
        {
            s += data[i];
        }
        return s;
    }

    float mean()
    {
        f32 s = this->sum() / cols;
        return s;
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
        M out = M::zeros(size, stride);
        for (u32 i = 0; i < size * stride; ++i)
        {
            out.data[i] = data[i] * data[i];
        }
        return out;
    }

    static M MatMul(M A, M B)
    {
        assert(A.stride == B.size);
        M Out = M::zeros(A.size, B.stride);
        for (u32 i = 0; i < A.size; i++)
        {
            for (u32 j = 0; j < B.stride; j++)
            {
                Out.data[i * B.stride + j] = 0;
                for (u32 k = 0; k < A.stride; k++)
                {
                    Out.data[i * B.stride + j] += A.data[i * A.stride + k] * B.data[k * B.stride + j];
                }
            }
        }
        return Out;
    }

    static void MatMul_(M A, M B, M &Out)
    {
        for (u32 i = 0; i < A.size; i++)
        {
            for (u32 j = 0; j < B.stride; j++)
            {
                Out.data[i * B.stride + j] = 0;
                for (u32 k = 0; k < A.stride; k++)
                {
                    Out.data[i * B.stride + j] += A.data[i * A.stride + k] * B.data[k * B.stride + j];
                }
            }
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
    static Layer *create(u32 input, u32 output)
    {
        // Allocate memory for the layer on the memory arena
        Layer *l = (Layer *)PushSize(&MemoryArena, sizeof(Layer));

#if GLOROT_UNIFORM
        // Initialize the weight matrix with Glorot uniform distribution
        l->setGlorotUniform(input, output);
#else
        l->setRandomUniform(-0.05, 0.05);
#endif
        // Initialize the weight and bias matrices with random values
        l->w = M::rand(input, output);
        l->b = M::zeros(1, output);

        // Initialize the gradient matrices to zero
        l->dw = M::zeros(input, output);
        l->db = M::zeros(1, output);
        return l;
    }

    // Basic constructor for the layer
    Layer(u32 input, u32 output) : w(M::rand(input, output)), b(M::zeros(1, output)) {}

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
            for (u32 j = 0; j < x.cols; ++j)
            {
                accum += x.data[j] * w.data[j * w.cols + i];
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
            f32 accum = 0;
            for (u32 l = 0; l < grad.cols; ++l)
            {
                accum += grad.data[l] * w.data[k * w.cols + l];
            }
            out.data[k] = accum;
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
    f32 *data = PushSize(&MemoryArena, data_size);
    char *buffer = reinterpret_cast<char *>(&data[0]);

    std::printf("Want to receive %zu", data_size);
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

void printMemoryInfo()
{
    std::printf("\nMemory used %lu\nMemory available %lu\n\n", MemoryArena.Used, MemoryArena.Size - MemoryArena.Used);
}

#define BEGIN_TRAIN_LOOP(m, epochs, lr, batch_size) \
    u32 _epochs_ = epochs;                          \
    u32 _m_ = m;                                    \
    f32 _lr_ = lr;                                  \
    u32 _batch_size_ = batch_size;                  \
    for (u32 epoch = 0; epoch < _epochs_; ++epoch)  \
    {                                               \
        f32 error = 0.0f;
        f32 valid_error = 0.0f;
#define END_TRAIN_LOOP                                                                                                                          \
    std::printf("Epoch[%i/%i] - Batch size: %u - Training Loss: %f - Valid Loss: %f - Learing rate: %f\n", epoch, _epochs_, _batch_size_, error / (f32)m, valid_error / (f32) m_valid ,_lr_); \
    }

int main()
{
    // Reserve memory arena which we'll use
    InitMemory(1024 * 1024 * 1024);

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

    Layer *l1 = Layer::create(29, 32);
    Layer *l2 = Layer::create(32, 20);
    Layer *l3 = Layer::create(20, 32);
    Layer *l4 = Layer::create(32, 29);

    u32 MemoryUsed = MemoryArena.Used;
    printMemoryInfo();

    // Hyperparameters
    u32 epochs = 100;
    f32 lr = 0.001;
    u32 m = 227451; // 227451
    u32 m_valid = 56962;
    u32 batchsize = 32;

    // BEGIN_TRAIN_LOOP(m-(data size), epochs, lr, batch_size)
    BEGIN_TRAIN_LOOP(m, epochs, lr, batchsize)
    {
        valid_error = 0.0f;
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
    }
    END_TRAIN_LOOP

    X_train[0].print();
    M a = Tanh(l1->forward(X_train[0]));
    M b = Tanh(l2->forward(a));
    M c = Tanh(l3->forward(b));
    M o = l4->forward(c);
    o.print();

    std::printf("Error: %f \n", Mse(X_train[0], o));
    free(MemoryArena.Base);
}