#include <cmath>
float sig(float x)
{
    return 1 / (1 + exp(-x));
}
float dot(float *v, float *w, unsigned int n)
{
    float sum = 0;
    for (unsigned int i = 0; i < n; i++)
        sum += v[i] * w[i];
    return sum;
}
class NN
{
public:
    NN(unsigned int input, unsigned int hidden, unsigned int output, float rate) : inputs(input), hidden_l(hidden), outputs(output), l_rate(rate)
    {
        this->h_weights = new float *[hidden];
        for (unsigned int i = 0; i < hidden; i++)
        {
            this->h_weights[i] = new float[input];
            for (unsigned int j = 0; j < input; j++)
            {
                float r = 1 / sqrt(hidden);
                this->h_weights[i][j] = (((float)rand()) / (float)RAND_MAX) * 2 * r - r;
            }
        }
        this->o_weights = new float *[output];
        for (unsigned int i = 0; i < output; i++)
        {
            this->o_weights[i] = new float[hidden];
            for (unsigned int j = 0; j < hidden; j++)
            {
                float r = 1 / sqrt(output);
                this->o_weights[i][j] = (((float)rand()) / (float)RAND_MAX) * 2 * r - r;
            }
        }
    }
    void train(float *input, unsigned int output)
    {
        float *h_out = new float[this->hidden_l];
        for (unsigned int i = 0; i < this->hidden_l; i++)
            h_out[i] = sig(dot(input, this->h_weights[i], this->inputs));
        float *o_out = new float[this->outputs];
        for (unsigned int j = 0; j < this->outputs; j++)
            o_out[j] = sig(dot(h_out, this->o_weights[j], this->hidden_l));
        float *o_error = new float[this->outputs];
        for (unsigned int j = 0; j < this->outputs; j++)
            o_error[j] = o_out[j];
        o_error[output]--;
        float *h_error = new float[this->hidden_l];
        for (unsigned int i = 0; i < this->hidden_l; i++)
            for (unsigned int j = 0; j < this->outputs; j++)
                h_error[i] += o_error[j] * this->o_weights[j][i];
        for (unsigned int i = 0; i < this->outputs; i++)
            for (unsigned int j = 0; j < this->hidden_l; j++)
                this->o_weights[i][j] -= (1 - o_out[i]) * o_out[i] * o_error[i] * h_out[j] * this->l_rate;
        for (unsigned int i = 0; i < this->hidden_l; i++)
            for (unsigned int j = 0; j < this->inputs; j++)
                this->h_weights[i][j] -= (1 - h_out[i]) * h_out[i] * h_error[i] * input[j] * this->l_rate;
    }
    unsigned int predict(float *input)
    {
        float *h_out = new float[this->hidden_l];
        for (unsigned int i = 0; i < this->hidden_l; i++)
            h_out[i] = sig(dot(input, this->h_weights[i], this->inputs));
        float max = dot(h_out, this->o_weights[0], this->hidden_l);
        unsigned int max_index = 0;
        for (unsigned int i = 1; i < this->outputs; i++)
        {
            float value = dot(h_out, this->o_weights[i], this->hidden_l);
            if (value > max)
            {
                max = value;
                max_index = i;
            }
        }
        return max_index;
    }
    ~NN()
    {
        for (unsigned int i = 0; i < this->hidden_l; i++)
            delete[] h_weights[i];
        delete[] h_weights;
        for (unsigned int i = 0; i < this->outputs; i++)
            delete[] o_weights[i];
        delete[] o_weights;
    }
private:
    const unsigned int inputs, hidden_l, outputs;
    const float l_rate;
    float **h_weights, **o_weights;
};