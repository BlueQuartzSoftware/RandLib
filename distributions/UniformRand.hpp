#pragma once

#include "UnivariateDistribution.hpp"

/**
 * @brief The UniformRand class <BR>
 * Uniform continuous distribution
 *
 * f(x | a, b) = 1 / (b - a) for a < x < b
 *
 * Notation: X ~ U(a, b)
 *
 * Related distributions: <BR>
 * X ~ B(1, 1, a, b) <BR>
 * (X - a) / (b - a) ~ IH(1)
 */
template <typename RealType = double>
class RANDLIB_EXPORT UniformRand : public UnivariateDistribution<RealType>
{
    static_assert(std::is_floating_point_v<RealType>, "Continuous distribution supports only floating-point types");

    RealType a = 0; ///< min bound
    RealType b = 1; ///< max bound
    RealType bma = 1; ///< b-a
    RealType bmaInv = 1; ///< 1/(b-a)
    RealType logbma = 0; ///< log(b-a)

    /**
     * @fn SetSupport
     * @param minValue a
     * @param maxValue b
     */
    void SetSupport(double minValue, double maxValue)
    {
        if(minValue >= maxValue)
            throw std::invalid_argument("Beta distribution: minimum value should be "
                                        "smaller than maximum value");

        a = minValue;
        b = maxValue;
        bma = b - a;
        bmaInv = 1.0 / bma;
        logbma = std::log(bma);
    }

public:
  UniformRand(double minValue = 0, double maxValue = 1)
: UnivariateDistribution<RealType>()
  {
      SetSupport(minValue,maxValue);
  }

  std::string Name() const override
  {
      return "Uniform(" + this->toStringWithPrecision(MinValue()) + ", " + this->toStringWithPrecision(MaxValue()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return FINITE_T;
  }
  RealType MinValue() const override
  {
    return this->a;
  }
  RealType MaxValue() const override
  {
    return this->b;
  }

  double f(const RealType& x) const override
  {
      return (x < this->a || x > this->b) ? 0.0 : this->bmaInv;
  }

  double logf(const RealType& x) const override
  {
      return (x < this->a || x > this->b) ? -INFINITY : -this->logbma;
  }

  double F(const RealType& x) const override
  {
      if(x < this->a)
          return 0.0;
      return (x > this->b) ? 1.0 : this->bmaInv * (x - this->a);
  }

  double S(const RealType& x) const override
  {
      if(x < this->a)
          return 1.0;
      return (x > this->b) ? 0.0 : this->bmaInv * (this->b - x);
  }

  RealType Variate() const override
  {
      return this->a + StandardVariate(this->localRandGenerator) * this->bma;
  }

  /**
   * @fn StandardVariate
   * @param randGenerator
   * @return a random number on interval (0,1) if no preprocessors are specified
   */
  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
#ifdef RANDLIB_UNIDBL
      /// generates this->a random number on [0,1) with 53-bit resolution, using 2 32-bit integer variate
  double x;
  unsigned int a, b;
  this->a = randGenerator.Variate() >> 6; /// Upper 26 bits
  b = randGenerator.Variate() >> 5;       /// Upper 27 bits
  x = (this->a * 134217728.0 + this->b) / 9007199254740992.0;
  return x;
#elif defined(RANDLIB_JLKISS64)
      /// generates this->a random number on [0,1) with 53-bit resolution, using 64-bit integer variate
  double x;
  unsigned long long this->a = randGenerator.Variate();
  this->a = (this->a >> 12) | 0x3FF0000000000000ULL; /// Take upper 52 bit
  *(reinterpret_cast<unsigned long long*>(&x)) = a;  /// Make this->a double from bits
  return x - 1.0;
#else
      RealType x = randGenerator.Variate();
      x += 0.5;
      x /= 4294967296.0;
      return x;
#endif
  }

  /**
   * @fn StandardVariateClosed
   * @param randGenerator
   * @return a random number on interval [0,1]
   */
  static RealType StandardVariateClosed(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      RealType x = randGenerator.Variate();
      return x / 4294967295.0;
  }

  /**
   * @fn StandardVariateHalfClosed
   * @param randGenerator
   * @return a random number on interval [0,1)
   */
  static RealType StandardVariateHalfClosed(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      RealType x = randGenerator.Variate();
      return x / 4294967296.0;
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
      for(RealType& var : outputData)
          var = this->Variate();
  }

  long double Mean() const override
  {
      return 0.5 * (this->b + this->a);
  }

  long double Variance() const override
  {
      return this->bma * this->bma / 12;
  }

  RealType Median() const override
  {
      return 0.5 * (this->b + this->a);
  }

    RealType Mode() const override
  {
      /// this can be any value in [a, b]
      return 0.5 * (this->b + this->a);
  }

  long double Skewness() const override
  {
      return 0.0;
  }

  long double ExcessKurtosis() const override
  {
      return -1.2;
  }

    long double Entropy() const
    {
        return (this->b == this->a) ? -INFINITY : std::log(this->bma);
    }

    double LikelihoodFunction(const std::vector<RealType>& sample) const override
    {
        bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(this->a, sample) && this->allElementsAreNotGreaterThan(this->b, sample);
        return sampleIsInsideInterval ? std::pow(this->bma, -sample.size()) : 0.0;
    }

    double LogLikelihoodFunction(const std::vector<RealType>& sample) const override
    {
        bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(this->a, sample) && this->allElementsAreNotGreaterThan(this->b, sample);
        int sample_size = sample.size();
        return sampleIsInsideInterval ? -sample_size * this->logbma : -INFINITY;
    }

    /**
     * @fn FitMinimum
     * fit minimum with maximum-likelihood estimator if unbiased == false,
     * fit minimum using UMVU estimator otherwise
     * @param sample
     */
    void FitMinimum(const std::vector<RealType>& sample, bool unbiased = false)
    {
        if(!this->allElementsAreNotGreaterThan(this->b, sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));
        RealType minVar = *std::min_element(sample.begin(), sample.end());

        if(unbiased == true)
        {
            int n = sample.size();
            /// E[min] = b - n / (n + 1) * (this->b - this->a)
            RealType minVarAdj = (minVar * (n + 1) - this->b) / n;
            if(!this->allElementsAreNotSmallerThan(minVarAdj, sample))
                throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_LARGE_A + this->toStringWithPrecision(minVarAdj)));
            SetSupport(minVarAdj, this->b);
        }
        else
        {
            SetSupport(minVar, this->b);
        }
    }

    /**
     * @fn FitMaximum
     * fit maximum with maximum-likelihood estimator if unbiased == false,
     * fit maximum using UMVU estimator otherwise
     * @param sample
     */
    void FitMaximum(const std::vector<RealType>& sample, bool unbiased = false)
    {
        if(!this->allElementsAreNotSmallerThan(this->a, sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
        RealType maxVar = *std::max_element(sample.begin(), sample.end());

        if(unbiased == true)
        {
            int n = sample.size();
            /// E[max] = (this->b - this->a) * n / (n + 1) + a
            RealType maxVarAdj = (maxVar * (n + 1) - this->a) / n;
            if(!this->allElementsAreNotGreaterThan(maxVarAdj, sample))
                throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_SMALL_B + this->toStringWithPrecision(maxVarAdj)));
            SetSupport(this->a, maxVarAdj);
        }
        else
        {
            SetSupport(this->a, maxVar);
        }
    }

    /**
     * @fn Fit
     * fit support with maximum-likelihood estimator if unbiased == false,
     * fit support using UMVU estimator otherwise
     * @param sample
     */
    void Fit(const std::vector<RealType>& sample, bool unbiased = false)
    {
        double minVar = *std::min_element(sample.begin(), sample.end());
        double maxVar = *std::max_element(sample.begin(), sample.end());
        if(unbiased == true)
        {
            int n = sample.size();
            /// E[min] = b - n / (n + 1) * (this->b - this->a)
            RealType minVarAdj = (minVar * n - maxVar) / (n - 1);
            /// E[max] = (this->b - this->a) * n / (n + 1) + a
            RealType maxVarAdj = (maxVar * n - minVar) / (n - 1);
            if(!this->allElementsAreNotSmallerThan(minVarAdj, sample))
                throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_LARGE_A + this->toStringWithPrecision(minVarAdj)));
            if(!this->allElementsAreNotGreaterThan(maxVarAdj, sample))
                throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_SMALL_B + this->toStringWithPrecision(maxVarAdj)));
            SetSupport(minVarAdj, maxVarAdj);
        }
        else
        {
            SetSupport(minVar, maxVar);
        }
    }

private:
  RealType quantileImpl(double p) const override
  {
      return this->a + this->bma * p;
  }

  RealType quantileImpl1m(double p) const override
  {
      return this->b - this->bma * p;
  }

  std::complex<double> CFImpl(double t) const override
  {
      double cosX = std::cos(t * this->b), sinX = std::sin(t * this->b);
      double cosY = std::cos(t * this->a), sinY = std::sin(t * this->a);
      std::complex<double> numerator(cosX - cosY, sinX - sinY);
      std::complex<double> denominator(0, t * this->bma);
      return numerator / denominator;
  }

  static constexpr char TOO_LARGE_A[] = "Minimum element of the sample is smaller than lower boundary returned by method: ";
  static constexpr char TOO_SMALL_B[] = "Maximum element of the sample is greater than upper boundary returned by method: ";
};
