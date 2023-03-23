#ifndef M4_HH
#define M4_HH
#include "M3.h"
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

    M4 operator+(M4 B)
    {
        M4 out = M4::zeros(d1, d2, d3, d4);
        for (u32 i = 0; i < d1 * d2 * d3 * d4; ++i)
        {
            out.data[i] = data[i] + B.data[i];
        }
        return out;
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

};
#endif