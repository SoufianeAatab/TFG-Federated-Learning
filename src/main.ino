#include "nn.h"

static Layer* inputLayer;  
static Layer* hiddenLayer; 
static Layer* outputLayer; 
static u32 memoryUsed = 0;

void setup(){
    initMemory(1024 * 32);

    inputLayer = Layer::create(1,64);
    hiddenLayer = Layer::create(64,64);
    outputLayer = Layer::create(64,1);

    memoryUsed = memoryArena.used;
    //Serial.println("Memory used for initialization: ");
    //Serial.print(memoryArena.used);

    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);

    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDB, HIGH);

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
    // reset memory arena
    memoryArena.used = memoryUsed;
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

void sendInt (int arg)
{
    // get access to the float as a byte-array:
    byte * data = (byte *) &arg; 

    // write the data to the serial
    Serial.write (data, sizeof (arg));
}

static f32 lr = 0.1f;
f32 train(f32 x, f32 y){
    int start = micros();
    M input = M(&x, 1,1);
    M target = M(&y, 1,1);

    M a = sigmoid(inputLayer->forward(input));
    M h = sigmoid(hiddenLayer->forward(a));
    M o = outputLayer->forward(h);

    M e3 = msePrime(target, o);
    M d3 = e3;

    M e2 = outputLayer->backward(d3);
    M d2 = e2 * sigmoidPrime(h);

    M e1 = hiddenLayer->backward(d2);
    M d1 = e1 * sigmoidPrime(a);

    outputLayer->updateWeights(d3, h, lr);
    hiddenLayer->updateWeights(d2, a, lr);
    inputLayer->updateWeights(d1, input, lr);
    sendInt(micros() - start);
    return mse(target, o);
}

f32 predict(f32 x){
    M input = M(&x, 1,1);
    M a = sigmoid(inputLayer->forward(input));
    M h = sigmoid(hiddenLayer->forward(a));
    M o = outputLayer->forward(h);

    return o.data[0];
}