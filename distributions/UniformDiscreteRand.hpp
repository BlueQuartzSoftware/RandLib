#pragma once

#include "BasicRandGenerator.hpp"
#include "UnivariateDistribution.hpp"

/**
 *@brief The DiscreteDistribution class <BR>
 * Abstract class for all discrete distributions
 */
template <typename IntType, class Engine = JLKiss64RandEngine>
class RANDLIB_EXPORT DiscreteDistribution : virtual public UnivariateDistribution<IntType, Engine>
{
  static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>, "Discrete distribution supports only signed integral types");

protected:
  DiscreteDistribution() = default;

  virtual ~DiscreteDistribution() = default;

public:
  //-------------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn logP
   * @param x
   * @return logarithm of probability to get x
   */
  virtual double logP(const IntType& x) const = 0;

  //-------------------------------------------------------------------------------------------
  // VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn P
   * @param k
   * @return probability to get k
   */
  virtual double P(const IntType& k) const
  {
    return std::exp(this->logP(k));
  }

  //-------------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn ProbabilityMassFunction
   * fill vector y with P(x)
   * @param x
   * @param y
   */
  void ProbabilityMassFunction(const std::vector<IntType>& x, std::vector<double>& y) const
  {
    for(size_t i = 0; i != x.size(); ++i)
      y[i] = this->P(x[i]);
  }

  /**
   * @fn LogProbabilityMassFunction
   * fill vector y with logP(x)
   * @param x
   * @param y
   */
  void LogProbabilityMassFunction(const std::vector<IntType>& x, std::vector<double>& y) const
  {
    for(size_t i = 0; i != x.size(); ++i)
      y[i] = this->logP(x[i]);
  }

  IntType Mode() const

      override
  {
    /// Works only for unimodal distributions
    IntType x = this->Median();
    double logProb = this->logP(x), newLogProb = this->logP(x + 1);
    if(logProb < newLogProb)
    {
      do
      {
        ++x;
        logProb = newLogProb;
        newLogProb = this->logP(x + 1);
      } while(logProb < newLogProb);
    }
    else
    {
      newLogProb = this->logP(x - 1);
      while(logProb < newLogProb)
      {
        --x;
        logProb = newLogProb;
        newLogProb = this->logP(x - 1);
      }
    }
    return x;
  }

  /**
   * @fn Hazard
   * @param x
   * @return hazard function
   */
  double Hazard(const IntType& x) const

      override
  {
    if(x < this->

           MinValue()

    )
      return 0.0; /// 0/1
    if(x > this->

           MaxValue()

    )
      return NAN; /// 0/0
    return this->P(x) / this->S(x);
  }

  /**
   * @fn LikelihoodFunction
   * @param sample
   * @return likelihood function of the distribution for given sample
   */
  double LikelihoodFunction(const std::vector<IntType>& sample) const

      override
  {
    long double res = 1.0;
    for(const IntType& var : sample)
      res *= this->P(var);
    return res;
  }

  /**
   * @fn LogLikelihoodFunction
   * @param sample
   * @return log-likelihood function of the distribution for given sample
   */
  double LogLikelihoodFunction(const std::vector<IntType>& sample) const

      override
  {
    long double res = 0.0;
    for(const IntType& var : sample)
      res += this->logP(var);
    return res;
  }

protected:
  IntType quantileImpl(double p, IntType initValue) const

      override
  {
    IntType down = initValue, up = down + 1;
    double fu = this->F(up), fd = this->F(down);
    /// go up
    while(fu < p)
    {
      fd = fu;
      fu = this->F(++up);
    }
    down = up - 1;
    /// go down
    while(fd > p)
    {
      fd = this->F(--down);
    }
    up = down + 1;
    /// if lower quantile is not equal probability, we return upper quantile
    return (fd < p) ? up : down;
  }

  IntType quantileImpl(double p) const

      override
  {
    /// We use quantile from sample as an initial guess
    static constexpr int SAMPLE_SIZE = 128;
    static std::vector<IntType> sample(SAMPLE_SIZE);
    this->Sample(sample);
    int index = p * SAMPLE_SIZE;
    if(index == 0)
      return this->quantileImpl(p, *std::min_element(sample.begin(), sample.end()));
    std::nth_element(sample.

                     begin(),
                     sample

                             .

                         begin()

                         + index,
                     sample.

                     end()

    );
    return this->quantileImpl(p, sample[index]);
  }

  IntType quantileImpl1m(double p, IntType initValue) const

      override
  {
    IntType down = initValue, up = down + 1;
    double su = this->S(up), sd = this->S(down);
    /// go up
    while(su > p)
    {
      sd = su;
      su = this->S(++up);
    }
    down = up - 1;
    /// go down
    while(sd < p)
    {
      sd = this->S(--down);
    }
    up = down + 1;

    /// if lower quantile is not equal probability, we return upper quantile
    return (sd > p) ? up : down;
  }

  IntType quantileImpl1m(double p) const

      override
  {
    /// We use quantile from sample as an initial guess
    static constexpr int SAMPLE_SIZE = 128;
    static std::vector<IntType> sample(SAMPLE_SIZE);
    this->Sample(sample);
    int index = p * SAMPLE_SIZE;
    if(index == 0)
      return this->quantileImpl1m(p, *std::max_element(sample.begin(), sample.end()));
    std::nth_element(sample.

                     begin(),
                     sample

                             .

                         begin()

                         + index,
                     sample.

                     end(),
                     std::greater<>()

    );
    return this->quantileImpl1m(p, sample[index]);
  }

  long double ExpectedValue(const std::function<double(IntType)>& funPtr, IntType minPoint, IntType maxPoint) const

      override
  {
    SUPPORT_TYPE suppType = this->SupportType();
    IntType k = minPoint, upperBoundary = maxPoint;
    if(suppType == FINITE_T || suppType == RIGHTSEMIFINITE_T)
    {
      k = std::max(k, this->MinValue());
    }
    if(suppType == FINITE_T || suppType == LEFTSEMIFINITE_T)
    {
      upperBoundary = std::min(upperBoundary, this->MaxValue());
    }

    double sum = 0;
    do
    {
      double addon = funPtr(k);
      if(addon != 0.0)
      {
        double prob = this->P(k);
        if(prob < MIN_POSITIVE)
          return sum;
        addon *= this->P(k);
        sum += addon;
      }
      ++k;
    } while(k <= upperBoundary);
    return sum;
  }
};

/**
 * @brief The UniformDiscreteRand class <BR>
 * Uniform discrete distribution
 *
 * Notation: X ~ U{a, ..., b}
 *
 * P(X = k) = 1 / (b - a + 1) for a <= k <= b
 */
template <typename IntType = int, class Engine = JLKiss64RandEngine>
class RANDLIB_EXPORT UniformDiscreteRand : public DiscreteDistribution<IntType, Engine>
{
  static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>, "Discrete distribution supports only signed integral types");

  size_t n = 1;                                                               ///< number of possible outcomes
  IntType a = 0;                                                              ///< min bound
  IntType b = 0;                                                              ///< max bound
  double nInv = 1;                                                            ///< 1/n
  double logN = 0;                                                            ///< log(n)
  unsigned long long MAX_RAND_UNBIASED = this->localRandGenerator.MaxValue(); ///< constant for unbiased generator

public:
  UniformDiscreteRand(IntType minValue = 0, IntType maxValue = 1)
  {
    SetBoundaries(minValue, maxValue);
  }

  ~UniformDiscreteRand() = default;

  std::string Name() const override
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

  double P(const IntType& k) const override
  {
    return (k < a || k > b) ? 0.0 : nInv;
  }

  double logP(const IntType& k) const override
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

  static IntType StandardVariate(IntType minValue = 0, IntType maxValue = 1, BasicRandGenerator<Engine>& randGenerator = ProbabilityDistribution<IntType, Engine>::staticRandGenerator)
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
