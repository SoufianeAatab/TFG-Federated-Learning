#define ONCOMPUTER 1
#define GLOROT_UNIFORM 1
#include "src/nn.h"

// Helper function to create one-hot encoded vectors
M one_hot(char c, int size) {
    M vec = M::zeros(1, size);
    vec.data[ c-'a'] = 1;
    return vec;
}

int main() {
    InitMemory(1024*1024*32);
    // Sample data: "hello"
    std::vector<char> input_data = {'h', 'e', 'l', 'l', 'o'};
    std::vector<char> target_data = {'e', 'l', 'l', 'o', 'h'};

    const int input_size = 26; // Number of unique characters (a-z)
    const int hidden_size = 10; // Size of the hidden layer
    const int output_size = 26; // Number of unique characters (a-z)
    const float lr = 0.01; // Learning rate

    // RNN* rnn = RNN::create(input_size, hidden_size, output_size);
    // u32 usedMem = MemoryArena.Used;

    // // Train the RNN
    // u32 epochs = 100;
    // for (int epoch = 0; epoch < epochs; ++epoch) {
    //     f32 error = 0.0f;
    //     for (size_t i = 0; i < input_data.size(); ++i) {
    //         M x = one_hot(input_data[i], input_size);
    //         M y = one_hot(target_data[i], output_size);

    //         M y_pred = rnn->forward(x);
    //         M dy = M::MatMul(CrossEntropyPrime(y, y_pred), SoftmaxPrime(y_pred));
    //         rnn->backward(x, dy, lr);

    //         error += CrossEntropy(y, y_pred);
    //         MemoryArena.Used = usedMem;
    //     }

    //     std::printf("[%u/%u] error: %f\n",epoch, epochs, error);
    // }

    // // Test the RNN
    // for (char c : input_data) {
    //     M x = one_hot(c, input_size);
    //     M y_pred = rnn->forward(x);
    //     int max_index = y_pred.argmax();
    //     char pred_char = 'a' + max_index;
    //     std::cout << c << " -> " << pred_char << std::endl;
    // }

    return 0;
}