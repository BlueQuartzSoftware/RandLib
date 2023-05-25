#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

namespace randlib
{
/**
 * @brief The CategoricalRand class <BR>
 *
 * P(X = k) = p_k for k = {0, ..., K-1}
 *
 * Notation: X ~ Cat(p_0, ..., p_{K-1})
 *
 * Related distributions: <BR>
 * X ~ Multin(1, p_0, ..., p_{K-1}) <BR>
 * If X ~ Bernoulli(p), then X ~ Cat(1 - p, p) <BR>
 * If X ~ Uniform-Discrete(0, K), then X ~ Cat(p, ..., p) with p = 1 / (K + 1)
 */
template <typename IntType = int>
class RANDLIB_EXPORT CategoricalRand : public randlib::DiscreteDistribution<IntType>
{
  std::vector<double> prob{1.0}; ///< vector of probabilities
  IntType K = 1;                 ///< number of possible outcomes

public:
  explicit CategoricalRand(std::vector<double>&& probabilities = {1.0})
  {
    SetProbabilities(std::move(probabilities));
  }

  String Name() const override
  {
    String str = "Categorical(";
    for(int i = 0; i != K - 1; ++i)
      str += this->toStringWithPrecision(prob[i]) + ", ";
    return str + this->toStringWithPrecision(prob[K - 1]) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  IntType MinValue() const override
  {
    return 0;
  }

  IntType MaxValue() const override
  {
    return K - 1;
  }

  void SetProbabilities(std::vector<double>&& probabilities)
  {
    if(probabilities.size() == 0 || std::accumulate(probabilities.begin(), probabilities.end(), 0.0) != 1.0)
      throw std::invalid_argument("Categorical distribution: probability parameters should sum to 1");
    else
    {
      prob = std::move(probabilities);
    }

    K = prob.size();
  }

  std::vector<double> GetProbabilities()
  {
    return prob;
  }

  double P(const IntType& k) const override
  {
    return (k < 0 || k >= K) ? 0.0 : prob[k];
  }

  double logP(const IntType& k) const override
  {
    return std::log(P(k));
  }

  double F(const IntType& k) const override
  {
    if(k < 0)
      return 0.0;
    if(k >= K)
      return 1.0;
    if(2 * k <= K)
    {
      double sum = 0.0;
      for(int i = 0; i <= k; ++i)
        sum += prob[i];
      return sum;
    }
    double sum = 1.0;
    for(int i = K - 1; i > k; --i)
      sum -= prob[i];
    return sum;
  }

  IntType Variate() const override
  {
    double U = UniformRand<double>::StandardVariate(this->localRandGenerator);
    return quantileImpl(U);
  }

  long double Mean() const override
  {
    long double sum = 0.0;
    for(int i = 1; i != K; ++i)
      sum += i * prob[i];
    return sum;
  }

  long double Variance() const override
  {
    long double mean = 0.0, secMom = 0.0;
    for(int i = 1; i != K; ++i)
    {
      long double aux = i * prob[i];
      mean += aux;
      secMom += i * aux;
    }
    return secMom - mean * mean;
  }

  IntType Mode() const override
  {
    auto maxProbIt = std::max_element(prob.begin(), prob.end());
    return std::distance(prob.begin(), maxProbIt);
  }

private:
  IntType quantileImpl(double p) const override
  {
    double sum = 0.0;
    for(IntType i = 0; i != K; ++i)
    {
      sum += prob[i];
      if(RandMath::areClose(sum, p) || sum > p)
        return i;
    }
    return K - 1;
  }

  IntType quantileImpl1m(double p) const override
  {
    double sum = 0;
    for(IntType i = K - 1; i >= 0; --i)
    {
      sum += prob[i];
      if(RandMath::areClose(sum, p) || sum > p)
        return i;
    }
    return 0.0;
  }

  std::complex<double> CFImpl(double t) const override
  {
    double re = 0.0;
    double im = 0.0;
    for(int i = 0; i != K; ++i)
    {
      re += prob[i] * std::cos(t * i);
      im += prob[i] * std::sin(t * i);
    }
    return std::complex<double>(re, im);
  }
};
} // namespace randlib
