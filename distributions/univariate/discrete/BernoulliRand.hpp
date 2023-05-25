#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ProbabilityDistribution.hpp"

#include "distributions/univariate/BasicRandGenerator.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"
#include "distributions/univariate/discrete/BinomialRand.hpp"

namespace randlib
{
/**
 * @brief The BernoulliRand class <BR>
 * Bernoulli distribution
 *
 * P(X = k) = p * 1_{k = 1} + (1 - p) * 1_{k = 0}
 *
 * Notation: X ~ Bernoulli(p)
 *
 * Related distributions: <BR>
 * X ~ Binomial(1, p) <BR>
 * X ~ Multin(1, 1 - p, p) <BR>
 * 2X - 1 ~ Rademacher
 */
class RANDLIB_EXPORT BernoulliRand : public BinomialDistribution<int>
{
  unsigned long long boundary = 0; ///< coefficient for faster random number generation

public:
  explicit BernoulliRand(double probability = 0.5)
  : BinomialDistribution(1, probability)
  {
    boundary = q * this->localRandGenerator.MaxValue();
  }

  String Name() const override
  {
    return "Bernoulli(" + this->toStringWithPrecision(GetProbability()) + ")";
  }

  void SetProbability(double probability)
  {
    if(probability < 0.0 || probability > 1.0)
      throw std::invalid_argument("Bernoulli distribution: probability parameter "
                                  "should in interval [0, 1]");
    SetParameters(1, probability);
    boundary = q * this->localRandGenerator.MaxValue();
  }

  double P(const int& k) const override
  {
    return (k == 0) ? q : ((k == 1) ? p : 0);
  }

  double logP(const int& k) const override
  {
    return (k == 0) ? log1mProb : ((k == 1) ? logProb : 0);
  }

  double F(const int& k) const override
  {
    return (k < 0) ? 0.0 : ((k < 1) ? q : 1);
  }

  double S(const int& k) const override
  {
    return (k < 0) ? 1.0 : ((k < 1) ? p : 0.0);
  }

  int Variate() const override
  {
    return this->localRandGenerator.Variate() > boundary;
  }

  static int Variate(double probability, RandGenerator& randGenerator = ProbabilityDistribution<int>::staticRandGenerator)
  {
    if(probability < 0.0 || probability > 1.0)
      throw std::invalid_argument("Bernoulli distribution: probability parameter "
                                  "should be in interval [0, 1]");
    return UniformRand<float>::StandardVariate(randGenerator) <= probability;
  }

  static int StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<int>::staticRandGenerator)
  {
    static const size_t maxDecimals = randGenerator.maxDecimals();
    static size_t decimals = 1;
    static unsigned long long X = 0;
    if(decimals == 1)
    {
      /// refresh
      decimals = maxDecimals;
      X = randGenerator.Variate();
    }
    else
    {
      --decimals;
      X >>= 1;
    }
    return X & 1;
  }

  void Sample(std::vector<int>& outputData) const override
  {
    if(p == 0.5)
    {
      for(int& var : outputData)
        var = StandardVariate(this->localRandGenerator);
    }
    else
    {
      for(int& var : outputData)
        var = this->Variate();
    }
  }

  inline double Entropy()
  {
    return -(p * logProb + q * log1mProb);
  }
};
} // namespace randlib
