#include "monte_carlo_integration.cuh"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "math_function.cuh"

#include <cuda_runtime.h>

#include <random>


static std::default_random_engine random_generator;
using std::cout;
using std::endl;
namespace {
     const int DEFAULT_NUM_SAMPLES = 1000;
}
MonteCarloIntegration::MonteCarloIntegration(MathFunction<double>& f)
: m_f(f),
  m_numSamples(DEFAULT_NUM_SAMPLES)
{
}
MonteCarloIntegration::MonteCarloIntegration(MathFunction<double>& f, int num_samples)
: m_f(f),
  m_numSamples(num_samples)



{
}
MonteCarloIntegration::MonteCarloIntegration(const MonteCarloIntegration& p)
: m_f(p.m_f),
  m_numSamples(p.m_numSamples)
{
}
MonteCarloIntegration::~MonteCarloIntegration()
{
}
MonteCarloIntegration& MonteCarloIntegration::operator =(const MonteCarloIntegration& p)
{
     if (this != &p)
     {
          m_f = p.m_f;
          m_numSamples = p.m_numSamples;
     }
     return *this;
}
void MonteCarloIntegration::setNumSamples(int n)
{
     m_numSamples = n;
}
double MonteCarloIntegration::integrateRegion(double a, double b, double min, double max) {
    // Allocate memory on GPU for results
    double* d_results;
    cudaMalloc(&d_results, m_numSamples * sizeof(double));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (m_numSamples + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = 1234ULL;
    bool positive = max > 0;

    monteCarloKernel<<<blocksPerGrid, threadsPerBlock>>>(d_results, m_numSamples, a, b, min, max, positive, d_f, seed);
    cudaDeviceSynchronize();

    // Copy results back to host
    double* h_results = new double[m_numSamples];
    cudaMemcpy(h_results, d_results, m_numSamples * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate points inside the region
    int pointsIn = 0;
    for (int i = 0; i < m_numSamples; ++i) {
        pointsIn += h_results[i];
    }

    // Free memory
    cudaFree(d_results);
    delete[] h_results;

    double percentageArea = pointsIn / double(m_numSamples);
    return (b - a) * (max - min) * percentageArea;
}

// Host-side getIntegral function
double MonteCarloIntegration::getIntegral(double a, double b) {
    // Find min and max of the function over the range
    std::uniform_real_distribution<> distrib(a, b);
    double max = 0;
    double min = 0;

    // Estimate min and max values
    for (int i = 0; i < m_numSamples; ++i) {
        double x = a + (b - a) * (i / double(m_numSamples));
        double y = d_f(x);
        if (y > max) max = y;
        if (y < min) min = y;
    }

    // Compute positive and negative integrals
    double positiveIntg = max > 0 ? integrateRegion(a, b, 0, max) : 0;
    double negativeIntg = min < 0 ? integrateRegion(a, b, min, 0) : 0;

    return positiveIntg - negativeIntg;
}

namespace  {
    class FSin : public MathFunction<double>
    {
    public:
        ~FSin();
        double operator()(double x);
    };
    FSin::~FSin()
    {
    }
    double FSin::operator()(double x)



    {
        return sin(x);
    }
}
int main()
{
     cout << "starting" << endl;
    FSin f;
    MonteCarloIntegration mci(f);
    double integral = mci.getIntegral(0.5, 4.9);
    cout << " the integral of the function is " << integral << endl;
     mci.setNumSamples(200000);
    integral = mci.getIntegral(0.5, 4.9);
    cout << " the integral of the function with 20000 intervals is " << integral << endl;
    return 0;
}



