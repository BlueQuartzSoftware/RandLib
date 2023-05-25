#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ProbabilityDistribution.hpp"

#include "distributions/univariate/discrete/NegativeBinomialRand.hpp"

namespace randlib
{
/**
 * @brief The GeometricRand class <BR>
 * Geometric distribution
 *
 * P(X = k) = p (1 - p)^k
 *
 * Notation: X ~ Geometric(p)
 *
 * Related distributions: <BR>
 * X ~ NB(1, p)
 */
template <typename IntType = int>
class RANDLIB_EXPORT GeometricRand : public PascalRand<IntType>
{
public:
  explicit GeometricRand(double probability = 0.5)
  : PascalRand<IntType>(1, probability)
  {
  }

  String Name() const override
  {
    return "Geometric(" + this->toStringWithPrecision(this->GetProbability()) + ")";
  }

  void SetProbability(double probability)
  {
    if(probability < 0.0 || probability > 1.0)
      throw std::invalid_argument("Geometric distribution: probability parameter "
                                  "should be in interval [0, 1]");
    this->SetParameters(1, probability);
  }

  double P(const IntType& k) const override
  {
    return (k < 0) ? 0 : this->p * std::exp(k * this->log1mProb);
  }

  double logP(const IntType& k) const override
  {
    return (k < 0) ? -INFINITY : this->logProb + k * this->log1mProb;
  }

  double F(const IntType& k) const override
  {
    return (k < 0) ? 0 : -std::expm1l((k + 1) * this->log1mProb);
  }

  double S(const IntType& k) const override
  {
    return (k < 0) ? 1 : std::exp((k + 1) * this->log1mProb);
  }

  IntType Variate() const override
  {
    typename PascalRand<IntType>::GENERATOR_ID genId = this->GetIdOfUsedGenerator();
    if(genId == this->EXPONENTIAL)
      return this->variateGeometricThroughExponential();
    if(genId == this->TABLE)
      return this->variateGeometricByTable();
    /// unexpected return
    throw std::runtime_error("Geometric distribution: sampling failed");
  }

  static IntType Variate(double probability, RandGenerator& randGenerator = ProbabilityDistribution<IntType>::staticRandGenerator)
  {
    if(probability > 1.0 || probability < 0.0)
      throw std::invalid_argument("Geometric distribution: probability parameter should be in interval "
                                  "[0, 1], but it's equal to " +
                                  std::to_string(probability));

    /// here we use 0.05 instead of 0.08 because log(q) wasn't hashed
    if(probability < 0.05)
    {
      double rate = -std::log1pl(-probability);
      float X = ExponentialRand<float>::StandardVariate(randGenerator) / rate;
      return std::floor(X);
    }

    double U = UniformRand<double>::StandardVariate(randGenerator);
    int x = 0;
    double prod = probability, sum = prod, qprob = 1.0 - probability;
    while(U > sum)
    {
      prod *= qprob;
      sum += prod;
      ++x;
    }
    return x;
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
    typename PascalRand<IntType>::GENERATOR_ID genId = this->GetIdOfUsedGenerator();
    if(genId == this->EXPONENTIAL)
    {
      for(IntType& var : outputData)
        var = this->variateGeometricThroughExponential();
    }
    else if(genId == this->TABLE)
    {
      for(IntType& var : outputData)
        var = this->variateGeometricByTable();
    }
  }

  IntType Median() const override
  {
    double median = -M_LN2 / this->log1mProb;
    double flooredMedian = std::floor(median);
    return (RandMath::areClose(median, flooredMedian, 1e-8)) ? flooredMedian - 1 : flooredMedian;
  }

  long double Entropy() const
  {
    double a = -this->q * this->log1mProb;
    double b = -this->p * this->logProb;
    return (a + b) / (M_LN2 * this->p);
  }
};
} // namespace randlib
