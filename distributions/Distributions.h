#ifndef RANDLIB_H
#define RANDLIB_H

#include "ProbabilityDistribution.h"
#include "distributions/univariate/BasicRandGenerator.h"

#include <complex>

namespace distributions {
/**
 * @brief The UnivariateDistribution class <BR>
 * Abstract class for all univariate probability distributions
 */
template <typename T>
class RANDLIBSHARED_EXPORT UnivariateDistribution
    : public ProbabilityDistribution<T> {
protected:
  UnivariateDistribution() = default;
  virtual ~UnivariateDistribution() = default;

public:
  /**
   * @fn SupportType
   * @return type of support
   */
  virtual SUPPORT_TYPE SupportType() const = 0;

  /**
   * @fn isLeftBounded
   * @return true if distribution is bounded from the left
   */
  bool isLeftBounded() const;

  /**
   * @fn isRightBounded
   * @return true if distribution is bounded from the right
   */
  bool isRightBounded() const;

  /**
   * @fn Mean
   * @return Mathematical expectation
   */
  virtual long double Mean() const = 0;

  /**
   * @fn Variance
   * @return Variance of random variable
   */
  virtual long double Variance() const = 0;

private:
  /**
   * @fn quantileImpl
   * @param p
   * @param initValue initial value of x
   * @return such x that F(x) = p
   */
  virtual T quantileImpl(double p, T initValue) const = 0;
  virtual T quantileImpl(double p) const = 0;

  /**
   * @fn quantileImpl1m
   * @param p
   * @param initValue initial value of x
   * @return such x that F(x) = 1 - p
   */
  virtual T quantileImpl1m(double p, T initValue) const = 0;
  virtual T quantileImpl1m(double p) const = 0;

protected:
  /**
   * @fn CFimpl
   * @param t
   * @return characteristic function implementation (inverse Fourier transform
   * of pdf)
   */
  virtual std::complex<double> CFImpl(double t) const;

  /**
   * @fn ExpectedValue
   * @param funPtr pointer on function g(x) with finite support which expected
   * value should be returned
   * @param minPoint min{x | g(x) ≠ 0}
   * @param maxPoint max{x | g(x) ≠ 0}
   * @return E[g(x)]
   */
  virtual long double ExpectedValue(const std::function<double(T)> &funPtr,
                                    T minPoint, T maxPoint) const = 0;

public:
  /**
   * @fn Quantile
   * @param p
   * @return quantileImpl(p) if p is in (0, 1)
   */
  T Quantile(double p) const;

  /**
   * @fn Quantile1m
   * @param p
   * @return quantileImpl1m(p) if p is in (0, 1)
   */
  T Quantile1m(double p) const;

  /**
   * @fn QuantileFunction
   * @param p
   * @param y
   * @return fills vector y with Quantile(p)
   */
  void QuantileFunction(const std::vector<double> &p, std::vector<T> &y);

  /**
   * @fn QuantileFunction1m
   * @param p
   * @param y
   * @return fills vector y with Quantile1m(p)
   */
  void QuantileFunction1m(const std::vector<double> &p, std::vector<T> &y);

  /**
   * @fn CF
   * @param t
   * @return CFImpl for t != 0 and 1 otherwise
   */
  std::complex<double> CF(double t) const;

  /**
   * @fn CharacteristicFunction
   * @param x input vector
   * @param y output vector: y = CF(x)
   */
  void CharacteristicFunction(const std::vector<double> &t,
                              std::vector<std::complex<double>> &y) const;

  /**
   * @fn Hazard
   * @param x input parameter
   * @return hazard function
   */
  virtual double Hazard(const T &x) const = 0;

  /**
   * @fn HazardFunction
   * @param x input vector
   * @param y output vector: y = Hazard(x)
   */
  void HazardFunction(const std::vector<T> &x, std::vector<double> &y) const;

  /**
   * @fn Median
   * @return such x that F(x) = 0.5
   */
  virtual T Median() const;

  /**
   * @fn Mode
   * Mode is the value, which has the largest probability to happen,
   * or, in the case of continuous distribution, has the largest value
   * of density function.
   * @return mode. In case, when it's not unique, return any of them.
   */
  virtual T Mode() const = 0;

  /**
   * @fn Skewness
   * @return E[((X - μ) / σ) ^ 3]
   * where μ is central moment and σ is standard deviation
   */
  virtual long double Skewness() const;

  /**
   * @fn Kurtosis
   * @return unbiased kurtosis = μ_4 / σ ^ 4
   */
  virtual long double Kurtosis() const;

  /**
   * @fn ExcessKurtosis
   * @return E[((X - μ) / σ) ^ 4]  - 3
   * (fourth moment around the mean divided by the square of the variance of the
   * probability distribution minus 3)
   */
  virtual long double ExcessKurtosis() const;

  /**
   * @fn SecondMoment
   * @return E[X^2]
   */
  virtual long double SecondMoment() const;

  /**
   * @fn ThirdMoment
   * @return E[X^3]
   */
  virtual long double ThirdMoment() const;

  /**
   * @fn FourthMoment
   * @return E[X^4]
   */
  virtual long double FourthMoment() const;

  /**
   * @fn LikelihoodFunction
   * @param sample
   * @return likelihood function for given sample
   */
  virtual double LikelihoodFunction(const std::vector<T> &sample) const = 0;

  /**
   * @fn LogLikelihoodFunction
   * @param sample
   * @return logarithm of likelihood function for given sample
   */
  virtual double LogLikelihoodFunction(const std::vector<T> &sample) const = 0;

protected:
  /**
   * @fn allElementsAreNotGreaterThan
   * @param value
   * @param sample
   * @return true if all elements in sample are not greater than given value
   */
  static bool allElementsAreNotGreaterThan(T value,
                                           const std::vector<T> &sample);

  /**
   * @fn allElementsAreNotSmallerThan
   * @param value
   * @param sample
   * @return true if all elements in sample are not smaller than given value
   */
  static bool allElementsAreNotSmallerThan(T value,
                                           const std::vector<T> &sample);

  /**
   * @fn allElementsAreNonNegative
   * @param sample
   * @return true if all elements in sample are non-negative
   */
  static bool allElementsAreNonNegative(const std::vector<T> &sample);

  /**
   * @fn allElementsArePositive
   * @param sample
   * @return true if all elements in sample are positive
   */
  static bool allElementsArePositive(const std::vector<T> &sample);

public:
  /**
   * @fn GetSampleSum
   * @param sample
   * @return sum of all elements in a sample
   */
  static long double GetSampleSum(const std::vector<T> &sample);

  /**
   * @fn GetSampleMean
   * @param sample
   * @return arithmetic average
   */
  static long double GetSampleMean(const std::vector<T> &sample);

  /**
   * @fn GetSampleLogMean
   * @param sample
   * @return arithmetic log-average
   */
  static long double GetSampleLogMean(const std::vector<T> &sample);

  /**
   * @fn GetSampleVariance
   * @param sample
   * @param mean known mean value
   * @return sample second central moment
   */
  static long double GetSampleVariance(const std::vector<T> &sample,
                                       double mean);

  /**
   * @fn GetSampleLogVariance
   * @param sample
   * @param logMean known log-mean value
   * @return sample log-variance
   */
  static long double GetSampleLogVariance(const std::vector<T> &sample,
                                          double logMean);

  /**
   * @fn GetSampleMeanAndVariance
   * @param sample
   * @return sample mean and variance
   */
  static LongDoublePair GetSampleMeanAndVariance(const std::vector<T> &sample);

  /**
   * @fn GetSampleLogMeanAndVariance
   * @param sample
   * @return sample log-mean and log-variance
   */
  static LongDoublePair
  GetSampleLogMeanAndVariance(const std::vector<T> &sample);

  /**
   * @fn GetSampleStatistics
   * @param sample
   * @return sample mean, variance, skewness and excess kurtosis
   */
  static std::tuple<long double, long double, long double, long double>
  GetSampleStatistics(const std::vector<T> &sample);
};

/**
 * @brief The BivariateDistribution class <BR>
 * Abstract class for all bivariate probability distributions
 */
template <class T1, class T2, typename T>
class RANDLIBSHARED_EXPORT BivariateDistribution
    : public ProbabilityDistribution<Pair<T>> {
protected:
  T1 X{}; ///< 1st marginal distributions
  T2 Y{}; ///< 2nd marginal distributions
  BivariateDistribution() = default;
  virtual ~BivariateDistribution() = default;

public:
  Pair<T> MinValue() const { return Pair<T>(X.MinValue(), Y.MinValue()); }
  Pair<T> MaxValue() const { return Pair<T>(X.MaxValue(), Y.MaxValue()); }

  void Reseed(unsigned long seed) const override;

  virtual LongDoublePair Mean() const final;
  virtual LongDoubleTriplet Covariance() const final;
  virtual long double Correlation() const = 0;
  virtual std::pair<T1, T2> GetMarginalDistributions() const final;
  virtual Pair<T> Mode() const = 0;
};

//-------------------------------------------------------------------------------------------
// Continuous
//-------------------------------------------------------------------------------------------
/**
 * @brief The ContinuousDistribution class <BR>
 * Abstract class for all continuous distributions
 */
template <typename RealType>
class RANDLIBSHARED_EXPORT ContinuousDistribution
    : virtual public UnivariateDistribution<RealType> {
  static_assert(std::is_floating_point_v<RealType>,
                "Continuous distribution supports only floating-point types");

protected:
  ContinuousDistribution() = default;
  virtual ~ContinuousDistribution() = default;

public:
  /**
   * @fn f
   * @param x
   * @return probability density function
   */
  virtual double f(const RealType &x) const { return std::exp(this->logf(x)); }

  /**
   * @fn logf
   * @param x
   * @return logarithm of probability density function
   */
  virtual double logf(const RealType &x) const = 0;

  /**
   * @fn ProbabilityDensityFunction
   * fill vector y with f(x)
   * @param x
   * @param y
   */
  void ProbabilityDensityFunction(const std::vector<RealType> &x,
                                  std::vector<double> &y) const;

  /**
   * @fn LogProbabilityDensityFunction
   * fill vector y with logf(x)
   * @param x
   * @param y
   */
  void LogProbabilityDensityFunction(const std::vector<RealType> &x,
                                     std::vector<double> &y) const;

  RealType Mode() const override;

protected:
  RealType quantileImpl(double p, RealType initValue) const override;
  RealType quantileImpl(double p) const override;
  RealType quantileImpl1m(double p, RealType initValue) const override;
  RealType quantileImpl1m(double p) const override;
  long double ExpectedValue(const std::function<double(RealType)> &funPtr,
                            RealType minPoint,
                            RealType maxPoint) const override;

public:
  double Hazard(const RealType &x) const override;
  double LikelihoodFunction(const std::vector<RealType> &sample) const override;
  double
  LogLikelihoodFunction(const std::vector<RealType> &sample) const override;

  /**
   * @fn KolmogorovSmirnovTest
   * @param orderStatistic sample sorted in ascending order
   * @param alpha level of test
   * @return true if sample is from this distribution according to asymptotic
   * KS-test, false otherwise
   */
  bool KolmogorovSmirnovTest(const std::vector<RealType> &orderStatistic,
                             double alpha) const;
};

/**
 * @brief The ContinuousBivariateDistribution class <BR>
 * Abstract class for all bivariate probability distributions
 * with marginal continuous distributions
 */
template <class T1, class T2, typename RealType>
class RANDLIBSHARED_EXPORT ContinuousBivariateDistribution
    : public BivariateDistribution<T1, T2, RealType> {
  static_assert(std::is_base_of_v<ContinuousDistribution<RealType>, T1>,
                "T1 must be a descendant of ContinuousDistribution");
  static_assert(std::is_base_of_v<ContinuousDistribution<RealType>, T2>,
                "T2 must be a descendant of ContinuousDistribution");

protected:
  ContinuousBivariateDistribution() = default;
  virtual ~ContinuousBivariateDistribution() = default;

public:
  virtual double f(const Pair<RealType> &point) const = 0;
  virtual double logf(const Pair<RealType> &point) const = 0;
};

/**
 * @brief The CircularDistribution class <BR>
 * Abstract class for all continuous circular distributions
 *
 * Note that all the moments are now useless, we implement circular moments
 * instead.
 */
template <typename RealType = double>
class RANDLIBSHARED_EXPORT CircularDistribution
    : public ContinuousDistribution<RealType> {
protected:
  double loc{};

  CircularDistribution(double location = 0);
  virtual ~CircularDistribution() {}

public:
  SUPPORT_TYPE SupportType() const override { return FINITE_T; }
  RealType MinValue() const override { return loc - M_PI; }
  RealType MaxValue() const override { return loc + M_PI; }

  void SetLocation(double location);
  inline double GetLocation() const { return loc; }

  long double Mean() const override { return NAN; }
  long double Variance() const override { return NAN; }
  long double Skewness() const override { return NAN; }
  long double ExcessKurtosis() const override { return NAN; }

  /**
   * @fn CircularMean
   * @return Circular mean of random variable
   */
  virtual long double CircularMean() const = 0;

  /**
   * @fn CircularVariance
   * @return Circular variance of random variable
   */
  virtual long double CircularVariance() const = 0;
};

//-------------------------------------------------------------------------------------------
// DISCRETE
//-------------------------------------------------------------------------------------------
/**
 *@brief The DiscreteDistribution class <BR>
 * Abstract class for all discrete distributions
 */
template <typename IntType>
class RANDLIBSHARED_EXPORT DiscreteDistribution
    : virtual public UnivariateDistribution<IntType> {
  static_assert(std::is_integral_v<IntType> && std::is_signed_v<IntType>,
                "Discrete distribution supports only signed integral types");

protected:
  DiscreteDistribution() = default;
  virtual ~DiscreteDistribution() = default;

public:
  /**
   * @fn P
   * @param k
   * @return probability to get k
   */
  virtual double P(const IntType &k) const { return std::exp(this->logP(k)); }

  /**
   * @fn logP
   * @param x
   * @return logarithm of probability to get x
   */
  virtual double logP(const IntType &x) const = 0;

  /**
   * @fn ProbabilityMassFunction
   * fill vector y with P(x)
   * @param x
   * @param y
   */
  void ProbabilityMassFunction(const std::vector<IntType> &x,
                               std::vector<double> &y) const;

  /**
   * @fn LogProbabilityMassFunction
   * fill vector y with logP(x)
   * @param x
   * @param y
   */
  void LogProbabilityMassFunction(const std::vector<IntType> &x,
                                  std::vector<double> &y) const;

  IntType Mode() const override;

protected:
  IntType quantileImpl(double p, IntType initValue) const override;
  IntType quantileImpl(double p) const override;
  IntType quantileImpl1m(double p, IntType initValue) const override;
  IntType quantileImpl1m(double p) const override;
  long double ExpectedValue(const std::function<double(IntType)> &funPtr,
                            IntType minPoint, IntType maxPoint) const override;

public:
  /**
   * @fn Hazard
   * @param x
   * @return hazard function
   */
  double Hazard(const IntType &x) const override;

  /**
   * @fn LikelihoodFunction
   * @param sample
   * @return likelihood function of the distribution for given sample
   */
  double LikelihoodFunction(const std::vector<IntType> &sample) const override;

  /**
   * @fn LogLikelihoodFunction
   * @param sample
   * @return log-likelihood function of the distribution for given sample
   */
  double
  LogLikelihoodFunction(const std::vector<IntType> &sample) const override;

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
  bool PearsonChiSquaredTest(const std::vector<IntType> &orderStatistic,
                             double alpha, int lowerBoundary, int upperBoundary,
                             size_t numberOfEstimatedParameters = 0) const;

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
  bool PearsonChiSquaredTest(const std::vector<IntType> &orderStatistic,
                             double alpha,
                             size_t numberOfEstimatedParameters = 0) const;
};

/**
 * @brief The DiscreteBivariateDistribution class <BR>
 * Abstract class for all bivariate probability distributions
 * with marginal discrete distributions
 */
template <class T1, class T2, typename IntType>
class RANDLIBSHARED_EXPORT DiscreteBivariateDistribution
    : public BivariateDistribution<T1, T2, IntType> {
  static_assert(std::is_base_of_v<DiscreteDistribution<IntType>, T1>,
                "T1 must be a descendant of DiscreteDistribution");
  static_assert(std::is_base_of_v<DiscreteDistribution<IntType>, T2>,
                "T2 must be a descendant of DiscreteDistribution");

protected:
  DiscreteBivariateDistribution() {}
  virtual ~DiscreteBivariateDistribution() {}

public:
  virtual double P(const Pair<IntType> &point) const = 0;
  virtual double logP(const Pair<IntType> &point) const = 0;
};

//-------------------------------------------------------------------------------------------
// SINGULAR
//-------------------------------------------------------------------------------------------
/**
 * @brief The SingularDistribution class <BR>
 * Abstract class for all singular distributions
 */
class RANDLIBSHARED_EXPORT SingularDistribution
    : public UnivariateDistribution<double> {
protected:
  SingularDistribution() = default;
  virtual ~SingularDistribution() = default;

private:
  double Hazard(const double &) const override;
  double Mode() const override;
  long double ExpectedValue(const std::function<double(double)> &funPtr,
                            double minPoint, double maxPoint) const override;
  double LikelihoodFunction(const std::vector<double> &sample) const override;
  double
  LogLikelihoodFunction(const std::vector<double> &sample) const override;
};
} // namespace distributions

#endif // RANDLIB_H
