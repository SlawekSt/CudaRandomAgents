#include "RandomNumberGenerator.h"
#include <omp.h>
thread_local std::random_device RandomNumberGenerator::rd;
thread_local std::mt19937 RandomNumberGenerator::rng(rd());

void getRandomNumbers(std::vector<float>& numbers, int size, float min, float max)
{
    #pragma omp parallel for    
    for (int i = 0; i < size; ++i)
    {
        numbers[i] = RandomNumberGenerator::randFloat(min, max);
    }
}
