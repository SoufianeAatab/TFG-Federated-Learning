#include "nn.h"

static Layer* l1;  
static Layer* l2; 
static Layer* l3; 
static u32 memoryUsed = 0;

void setup(){
    InitMemory(1024 * 32);

    l1 = Layer::Create(1,8);
    l2 = Layer::Create(8,8);
    l3 = Layer::Create(8,1);

    memoryUsed = MemoryArena.Used;
    Serial.println("Memory used for NN initialization: ");
    Serial.print(MemoryArena.Used);

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);

    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

    Serial.write("123");
}

void loop(){
    if(Serial.available() > 0){
        int inByte = Serial.read();
        switch(inByte){
            case 'r':
                digitalWrite(LEDR, LOW);
                digitalWrite(LEDG, HIGH);
                digitalWrite(LEDB, HIGH);
                break;
            case 'g':
                digitalWrite(LEDR, HIGH);
                digitalWrite(LEDG, LOW);
                digitalWrite(LEDB, HIGH);
                break;
            case 'b':
                digitalWrite(LEDR, HIGH);
                digitalWrite(LEDG, HIGH);
                digitalWrite(LEDB, LOW);
                break;
            case 't':
            {
                f32 x = readFloat();
                f32 y = readFloat();    
                f32 loss = train(x, y);
                sendFloat(loss);
                Serial.flush();

            }break;
            case 'p':
            {
                f32 x = readFloat();
                f32 y = predict(x);
                sendFloat(y);
                Serial.flush();
            }
            break;
        }
    }

    MemoryArena.Used = memoryUsed;
}

float readFloat() {
    byte res[4];
    while(Serial.available() < 4) {}
    for (int n = 0; n < 4; n++) {
        res[n] = Serial.read();
    }
    return *(float *)&res;
}

void sendFloat (float arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

static f32 lr = 0.1f;
f32 train(f32 x, f32 y){
    M input = M(&x, 1,1);
    M target = M(&y, 1,1);

    M a = Sigmoid(l1->forward(input));
    M h = Sigmoid(l2->forward(a));
    M o = l3->forward(h);

    M e3 = Loss(target, o);
    M d3 = e3;

    M e2 = l3->backward(d3);
    M d2 = e2 * SigmoidPrime(h);

    M e1 = l2->backward(d2);
    M d1 = e1 * SigmoidPrime(a);

    l3->UpdateWeights(d3, h, lr);
    l2->UpdateWeights(d2, a, lr);
    l1->UpdateWeights(d1, input, lr);

    return Mse(target, o);
}

f32 predict(f32 x){
    M input = M(&x, 1,1);
    M a = Sigmoid(l1->forward(input));
    M h = Sigmoid(l2->forward(a));
    M o = l3->forward(h);

    return o.data[0];
}