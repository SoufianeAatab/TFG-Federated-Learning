
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
