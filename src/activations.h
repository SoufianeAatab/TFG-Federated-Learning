#ifndef ACTIVATIONS_HH
#define ACTIVATIONS_HH

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

#endif