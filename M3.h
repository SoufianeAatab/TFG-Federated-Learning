#ifndef M3_HH
#define M3_HH
#include "M.h"
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
            o.data[i] = distribution(generator);
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
        std::printf("SHAPE: %d, %d, %d\n",d1,d2,d3 );
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

};
#endif
