#ifndef M_HH
#define M_HH
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
};
#endif
