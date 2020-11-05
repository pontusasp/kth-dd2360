#include <math.h>
#define N 64

// A scaling function to convert integers 0,1,...,N-1 to evenly spaced floats
float scale(int i, int n)
{
    return ((float)i) / (n - 1);
}

// Compute the distance between 2 points on a line.
float distance(float x1, float x2)
{
    return sqrt((x2 - x1) * (x2 - x1));
}

int main()
{
    float out[N] = {0.0};

    // Choose a reference value from which distances are measured.
    const float ref = 0.5;
    for(int i = 0; i < N; i++)
    {
        float x = scale(i, N);
        out[i] = distance(x, ref);
    }

    return 0;
}