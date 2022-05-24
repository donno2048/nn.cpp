#pragma once
#include <random>
class NN {
public:
    NN(unsigned int input, unsigned int hidden, unsigned int output, float rate);
    ~NN();
    void train(float* input, unsigned int output);
    unsigned int predict(float* input);
private:
    const unsigned int inputs;
    const unsigned int hidden_l;
    const unsigned int outputs;
    const float l_rate;
    float** h_weights;
    float** o_weights;
};