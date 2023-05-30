#pragma once

#include "BasicRandGenerator.h"
#include "UnivariateDistribution.h"

/**
 * @brief The UniformDiscreteRand class <BR>
 * Uniform discrete distribution
 *
 * Notation: X ~ U{a, ..., b}
 *
 * P(X = k) = 1 / (b - a + 1) for a <= k <= b
 */
template <typename IntType = int>
class RANDLIBSHARED_EXPORT UniformDiscreteRand : public UnivariateDistribution<IntType>
{
    static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>, "Discrete distribution supports only signed integral types");

  size_t n = 1;                                                               ///< number of possible outcomes
  IntType a = 0;                                                              ///< min bound
  IntType b = 0;                                                              ///< max bound
  double nInv = 1;                                                            ///< 1/n
  double logN = 0;                                                            ///< log(n)
  unsigned long long MAX_RAND_UNBIASED = this->localRandGenerator.MaxValue(); ///< constant for unbiased generator

protected:
    ~UniformDiscreteRand() = default;

public:
  UniformDiscreteRand(IntType minValue = 0, IntType maxValue = 1)
  {
      SetBoundaries(minValue, maxValue);
  }

  String Name() const override
  {
      return "Uniform Discrete(" + this->toStringWithPrecision(MinValue()) + ", " + this->toStringWithPrecision(MaxValue()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return FINITE_T;
  }

  IntType MinValue() const override
  {
    return a;
  }

  IntType MaxValue() const override
  {
    return b;
  }

  void SetBoundaries(IntType minValue, IntType maxValue)
  {
      if(minValue > maxValue)
          throw std::invalid_argument("Uniform discrete distribution: minimum value shouldn't be greater than maximum value");

      a = minValue;
      b = maxValue;

      n = b - a + 1;
      nInv = 1.0 / n;
      logN = std::log(n);

      unsigned long long MAX_RAND = this->localRandGenerator.MaxValue();
      MAX_RAND_UNBIASED = MAX_RAND - MAX_RAND % n - 1;
  }

  double P(const IntType& k) const
  {
      return (k < a || k > b) ? 0.0 : nInv;
  }

  double logP(const IntType& k) const
  {
      return (k < a || k > b) ? -INFINITY : -logN;
  }

  double F(const IntType& k) const override
  {
      if(k < a)
          return 0.0;
      if(k > b)
          return 1.0;
      return (k - a + 1) * nInv;
  }

  IntType Variate() const override
  {
      unsigned long intVar;
      do
      {
          intVar = this->localRandGenerator.Variate();
      } while(intVar > MAX_RAND_UNBIASED);
      return a + (intVar % n);
  }

  static IntType StandardVariate(IntType minValue = 0, IntType maxValue = 1, RandGenerator& randGenerator = ProbabilityDistribution<IntType>::staticRandGenerator)
  {
      unsigned long long MAX_RAND = randGenerator.MaxValue();
      IntType n = maxValue - minValue + 1;
      if(n <= 1)
          return minValue;
      unsigned long long MAX_RAND_UNBIASED = MAX_RAND - MAX_RAND % n - 1;
      unsigned long intVar;
      do
      {
          intVar = randGenerator.Variate();
      } while(intVar > MAX_RAND_UNBIASED);
      return minValue + (intVar % n);
  }

  long double Mean() const override
  {
      return 0.5 * (b + a);
  }

  long double Variance() const override
  {
      double nm1 = n - 1;
      double np1 = n + 1;
      return nm1 * np1 / 12;
  }

  IntType Median() const override
  {
      return (b + a) >> 1;
  }

  IntType Mode() const override
  {
      /// this can be any value in [a, b]
      return 0.5 * (a + b);
  }

  long double Skewness() const override
  {
      return 0.0;
  }

  long double ExcessKurtosis() const override
  {
      double kurt = n;
      kurt *= n;
      --kurt;
      kurt = 2.0 / kurt;
      ++kurt;
      return -1.2 * kurt;
  }

    long double Entropy() const
    {
        return logN;
    }

    double LikelihoodFunction(const std::vector<IntType>& sample) const override
    {
        bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(a, sample) && this->allElementsAreNotGreaterThan(b, sample);
        return sampleIsInsideInterval ? std::pow(n, -sample.size()) : 0.0;
    }

    double LogLikelihoodFunction(const std::vector<IntType>& sample) const override
    {
        bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(a, sample) && this->allElementsAreNotGreaterThan(b, sample);
        int sample_size = sample.size();
        return sampleIsInsideInterval ? -sample_size * logN : -INFINITY;
    }

    /**
     * @fn Fit
     * fit bounds via maximum-likelihood method
     * @param sample
     */
    void Fit(const std::vector<IntType>& sample)
    {
        IntType minVar = *std::min_element(sample.begin(), sample.end());
        IntType maxVar = *std::max_element(sample.begin(), sample.end());
        this->SetBoundaries(minVar, maxVar);
    }

private:
  std::complex<double> CFImpl(double t) const override
  {
      double at = a * t;
      double bp1t = (b + 1) * t;
      double reNum = std::cos(at) - std::cos(bp1t);
      double imNum = std::sin(at) - std::sin(bp1t);
      std::complex<double> numerator(reNum, imNum);
      std::complex<double> denominator(1.0 - std::cos(t), -std::sin(t));
      return nInv * numerator / denominator;
  }
};
