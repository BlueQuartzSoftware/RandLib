#pragma once

#include "distributions/univariate/continuous/NoncentralChiSquaredRand.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "math/RandMath.hpp"

#include "RandLib_export.hpp"

#include "RandLib.hpp"

#include <complex>

namespace randlib
{
/**
 *@brief The DiscreteDistribution class <BR>
 * Abstract class for all discrete distributions
 */
template <typename IntType>
class RANDLIB_EXPORT DiscreteDistribution

: virtual public UnivariateDistribution<IntType>
{
  static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>, "Discrete distribution supports only signed integral types");

protected:
  DiscreteDistribution() = default;

  virtual ~

      DiscreteDistribution() = default;

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

  /**
   * @fn PearsonChiSquaredTest
   * @param orderStatistic sample sorted in ascending order
   * @param alpha significance level of the test
   * @param lowerBoundary setting of the left most interval (-infinity,
   * lowerBoundary]
   * @param upperBoundary setting of the right most interval [upperBoundary,
   * infinity)
   * @param numberOfEstimatedParameters zero by default
   * @return true if sample is from this distribution according to Pearson's
   * chi-squared test, false otherwise
   *
   * Chi-squared test is implemented only for discrete distribution, as it is
   * much more complicated for continuous one. The problem is ambiguity of
   * grouping sample into intervals. In the case when parameters are estimated
   * by maximum-likelihood estimator, using original observations, statistics
   * might not follow asymptotic chi-square distribution and that leads to
   * serious underestimate of the error of the first kind. For more details
   * look: "The use of MLE in chi-square tests for goodness of fit" by Herman
   * Chernoff and E.L. Lehmann
   */
  bool PearsonChiSquaredTest(const std::vector<IntType>& orderStatistic, double alpha, int lowerBoundary, int upperBoundary, size_t numberOfEstimatedParameters = 0) const
  {
    size_t n = orderStatistic.size(), i = 0, k = 0;
    double nInv = 1.0 / n, sum = 0.0;

    /// Sanity checks
    if(lowerBoundary >= upperBoundary)
      throw std::invalid_argument("Lower boundary should be smaller than upper one");
    for(size_t j = 1; j != n; ++j)
    {
      if(orderStatistic[i] < orderStatistic[j - 1])
        throw std::invalid_argument("Order statistic should be sorted in ascending order");
    }
    if(orderStatistic[0] < this->MinValue())
      throw std::invalid_argument("Some elements in the sample are too small to belong to this "
                                  "distribution, they should be greater than " +
                                  this->toStringWithPrecision(this->MinValue()));
    if(orderStatistic[n - 1] > this->MaxValue())
      throw std::invalid_argument("Some elements in the sample are too large to belong to this "
                                  "distribution, they should be smaller than " +
                                  this->toStringWithPrecision(this->MaxValue()));

    /// Lower interval
    IntType x = orderStatistic[0];
    if(lowerBoundary > this->MinValue())
    {
      auto upIt = std::upper_bound(orderStatistic.begin(), orderStatistic.end(), lowerBoundary);
      i += upIt - orderStatistic.begin();
      x = orderStatistic[i];
      double prob = nInv * i, expectedProb = this->F(lowerBoundary);
      double addon = prob - expectedProb;
      addon *= addon;
      addon /= expectedProb;
      sum += addon;
      ++k;
    }
    /// Middle intervals
    while(i < n && x < upperBoundary)
    {
      size_t count = 1;
      x = orderStatistic[i];
      while(i + count < n && x == orderStatistic[i + count])
        ++count;
      double prob = nInv * count, expectedProb = this->P(x);
      double addon = prob - expectedProb;
      addon *= addon;
      addon /= expectedProb;
      sum += addon;
      i += count;
      ++k;
    }
    /// Upper interval
    if(upperBoundary < this->MaxValue())
    {
      double prob = nInv * (n - i), expectedProb = this->S(upperBoundary);
      double addon = prob - expectedProb;
      addon *= addon;
      addon /= expectedProb;
      sum += addon;
      ++k;
    }

    if(k <= numberOfEstimatedParameters + 1)
    {
      throw std::invalid_argument("Sample is too small, number of groups (" + this->toStringWithPrecision(k) + ") should be bigger than number of estimated parameters plus 1 (" +
                                  this->toStringWithPrecision(numberOfEstimatedParameters + 1) + ")");
    }
    double statistic = n * sum;

    ChiSquaredRand<IntType> X = ChiSquaredRand<IntType>(k - 1);
    double q = X.Quantile1m(alpha);
    return statistic <= q;
  }

  /**
   * @fn PearsonChiSquaredTest
   * @param orderStatistic sample sorted in ascending order
   * @param alpha significance level of the test
   * @param numberOfEstimatedParameters zero by default
   * @return true if sample is from this distribution according to Pearson's
   * chi-squared test, false otherwise In this function user won't set upper and
   * lower intervals for tails. However it might be useful to group rare events
   * for chi-squared test to achieve better results
   */
  bool PearsonChiSquaredTest(const std::vector<IntType>& orderStatistic, double alpha, size_t numberOfEstimatedParameters = 0) const
  {
    return PearsonChiSquaredTest(orderStatistic, alpha, this->MinValue(), this->MaxValue(), numberOfEstimatedParameters);
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
 * @brief The DiscreteBivariateDistribution class <BR>
 * Abstract class for all bivariate probability distributions
 * with marginal discrete distributions
 */
template <class T1, class T2, typename IntType>
class RANDLIB_EXPORT DiscreteBivariateDistribution

: public BivariateDistribution<T1, T2, IntType>
{
  static_assert(std::is_base_of_v<DiscreteDistribution<IntType>, T1>, "T1 must be a descendant of DiscreteDistribution");
  static_assert(std::is_base_of_v<DiscreteDistribution<IntType>, T2>, "T2 must be a descendant of DiscreteDistribution");

protected:
  DiscreteBivariateDistribution()
  {
  }

  virtual ~DiscreteBivariateDistribution()
  {
  }

public:
  //-------------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //-------------------------------------------------------------------------------------------

  virtual double P(const Pair<IntType>& point) const = 0;

  virtual double logP(const Pair<IntType>& point) const = 0;
};
} // namespace randlib
