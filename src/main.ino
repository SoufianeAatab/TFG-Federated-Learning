#define ONCOMPUTER 0
#include "nn.h"

static Layer* l1;  
static Layer* l2; 
static Layer* l3; 
static u32 memoryUsed = 0;
static f32 lr = 0.01f;

struct train_data{
    M input;
    M target;
};

#define INPUT_SIZE 64
#define OUTPUT_SIZE 10

void setup(){
    Serial.begin(9600);
    InitMemory(1024 * 32 * 4);

    l1 = Layer::create(INPUT_SIZE,32);
    l2 = Layer::create(32,32);
    l3 = Layer::create(32,OUTPUT_SIZE);

    memoryUsed = MemoryArena.Used;

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);

    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    // put your setup code here, to run once:
    randomSeed(0);

    initNetworkModel();
    digitalWrite(LED_BUILTIN, LOW);    // OFF   
}

void read_weights(Layer* l) {
    //Serial.println("Receiving weights..");
    float* weights = l->w.data;

    for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
        Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        weights[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->w.rows * l->w.cols; ++i) {
    //     Serial.print(l->w.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->w.std());
}

void read_bias(Layer* l) {
    //Serial.println("Receiving bias..");
    float* bias = l->b.data;
    for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
        Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        bias[i] = *reinterpret_cast<float*>(bytes);
    }

    // for (uint16_t i = 0; i < l->b.rows * l->b.cols; ++i) {
    //     Serial.print(l->b.data[i]);
    //     Serial.print(" ");
    // }
    // Serial.print(l->b.std());
}

void sendLayerMetaData(Layer* l){
    sendInt(l->w.rows);
    sendInt(l->w.cols);
}

void initNetworkModel(){
    Serial.println("Start receiving model");
    char signal;
    do {
        signal = Serial.read();
        Serial.println("Waiting for new model...");
    } while(signal != 's'); // s -> START

    Serial.println("start");
    Serial.write("i");
    // How many layers?
    sendInt(3);
    sendLayerMetaData(l1);
    sendLayerMetaData(l2);
    sendLayerMetaData(l3);

    // TODO : Change to read_weights(l1): this will read bias and weights within the same function.
    read_bias(l1);
    read_weights(l1);

    read_bias(l2);
    read_weights(l2);

    read_bias(l3);
    read_weights(l3);
}

train_data receive_sample(){
    train_data data = {};
    data.input = M::zeros(1,INPUT_SIZE);
    data.target = M::zeros(1, OUTPUT_SIZE);

    f32* input = data.input.data;
    f32* target = data.target.data;

    for(u16 i=0;i<INPUT_SIZE;++i){
        Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
    }
    
    for(u16 i=0;i<OUTPUT_SIZE;++i){
        Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        target[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}

M receive_sample_inference(){
    M data = M::zeros(1,INPUT_SIZE);
    f32* input = data.data;

    for(u16 i=0;i<INPUT_SIZE;++i){
        Serial.write('n');
        while (Serial.available() < 4);

        char bytes[4];
        Serial.readBytes(bytes, 4);
        input[i] = *reinterpret_cast<float*>(bytes);
    }

    return data;
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

void sendM(M arg)
{
    for(u32 i=0;i<arg.cols * arg.rows;++i){
        Serial.write('n');
        sendFloat(arg.data[i]);
    }
}

void sendFloat (float arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

void sendInt (int arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

f32 train(M input, M target){
    // l1->resetGradients();
    // l2->resetGradients();
    // l3->resetGradients();

    M a = Sigmoid(l1->forward(input));
    M b = Sigmoid(l2->forward(a));
    M o = Softmax(l3->forward(b));

    M e3 = M::MatMul(CrossEntropyPrime(target, o), SoftmaxPrime(o));
    M e2 = l3->backward(e3) * SigmoidPrime(b);
    M e1 = l2->backward(e2) * SigmoidPrime(a);

    l3->dw = l3->getDelta(e3, b);
    l2->dw = l2->getDelta(e2, a);
    l1->dw = l1->getDelta(e1, input);

    l3->db = e3;
    l2->db = e2;
    l1->db = e1;

    l3->UpdateWeights(lr);
    l2->UpdateWeights(lr);
    l1->UpdateWeights(lr);

    // l1->UpdateWeights(e1, input, lr);
    // l2->UpdateWeights(e2, a, lr);
    // l3->UpdateWeights(e3, b, lr);

    return CrossEntropy(target, o);
}

M predict(M input){
    M a = Sigmoid(l1->forward(input));
    M b = Sigmoid(l2->forward(a));
    M o = Softmax(l3->forward(b));

    return o;
}

void loop(){
    if(Serial.available() > 0){
        char inByte = Serial.read();
        if(inByte == 't'){
            train_data sample = receive_sample();
            f32 error = train(sample.input, sample.target);
            sendFloat(error);
        } else if(inByte == 'p'){
            M input = receive_sample_inference();
            M output = predict(input);
            sendM(output);
        }
    }
    MemoryArena.Used = memoryUsed;
}