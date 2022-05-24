#include "nn.h"
#include <iostream>
using namespace std;
int main(void) {
    NN nn(2, 2, 2, .2); // 2 inputs, 2 hidden, 2 outputs, learning rate .2
    float input[2] = {0, 0};
    int output = 0;
    for (int i = 0; i < 10000; i++) {
        input[0] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        input[1] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        output = input[0] + input[1] > 0 ? 1 : 0;
        nn.train(input, output);
    }
    for (int i = 0; i < 10; i++) {
        input[0] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        input[1] = (((float)rand()) / (float)RAND_MAX) * 2 - 1;
        output = input[0] + input[1] > 0 ? 1 : 0;
        cout << nn.predict(input) << " " << output << endl;
    }
    return 0;
}