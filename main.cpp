#include <iostream>
using namespace std;
#include <ctime>
#include "nn.h"
int main(void) {
    srand(time(NULL));
    NN nn(2, 2, 2, .2); // 2 inputs, 2 hidden, 2 outputs, learning rate .2
    float input[2] = {0, 0};
    unsigned int output = 0;
    for (unsigned int i = 0; i < 1e5; i++) {
        input[0] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        input[1] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        output = (input[0] + input[1]) > 0 ? 1 : 0;
        nn.train(input, output);
    }
    unsigned int successes = 0;
    for (unsigned int i = 0; i < 10000; i++) {
        input[0] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        input[1] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        output = (input[0] + input[1]) > 0 ? 1 : 0;
        if (nn.predict(input) == output) successes++;
    }
    cout << successes / 100. << "% Successes" << endl;
    return 0;
}