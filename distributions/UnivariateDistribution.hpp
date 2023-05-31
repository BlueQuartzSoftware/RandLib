#pragma once

#include "ProbabilityDistribution.hpp"

#include <complex>
#include <functional>
#include <numeric>

enum SUPPORT_TYPE
{
  FINITE_T,
  RIGHTSEMIFINITE_T,
  LEFTSEMIFINITE_T
};

/**
 * @brief The UnivariateDistribution class <BR>
 * Abstract class for all univariate probability distributions
 */
template <typename T, class Engine = JLKiss64RandEngine>
class  UnivariateDistribution : public ProbabilityDistribution<T, Engine>
{
protected:
  UnivariateDistribution() = default;
  virtual ~UnivariateDistribution() = default;

public:
  //--------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn SupportType
   * @return type of support
   */
  virtual SUPPORT_TYPE SupportType() const = 0;

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

  /**
   * @fn Hazard
   * @param x input parameter
   * @return hazard function
   */
  virtual double Hazard(const T& x) const = 0;

  /**
   * @fn Mode
   * Mode is the value, which has the largest probability to happen,
   * or, in the case of continuous distribution, has the largest value
   * of density function.
   * @return mode. In case, when it's not unique, return any of them.
   */
  virtual T Mode() const = 0;

  /**
   * @fn LikelihoodFunction
   * @param sample
   * @return likelihood function for given sample
   */
  virtual double LikelihoodFunction(const std::vector<T>& sample) const = 0;

  /**
   * @fn LogLikelihoodFunction
   * @param sample
   * @return logarithm of likelihood function for given sample
   */
  virtual double LogLikelihoodFunction(const std::vector<T>& sample) const = 0;

  //--------------------------------------------------------------------------------------
  // VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn Median
   * @return such x that F(x) = 0.5
   */
  virtual T Median() const
  {
    return quantileImpl(0.5);
  }

  /**
   * @fn Skewness
   * @return E[((X - μ) / σ) ^ 3]
   * where μ is central moment and σ is standard deviation
   */
  virtual long double Skewness() const
  {
    long double var = this->Variance();
    if(!std::isfinite(var))
      return NAN;
    long double mu = this->Mean(); /// var is finite, so is mu

    long double sum = this->ExpectedValue(
        [mu](double x) {
          double xmmu = x - mu;
          double skewness = xmmu * xmmu * xmmu;
          return skewness;
        },
        this->MinValue(), this->MaxValue());

    return sum / std::pow(var, 1.5);
  }

  /**
   * @fn ExcessKurtosis
   * @return E[((X - μ) / σ) ^ 4]  - 3
   * (fourth moment around the mean divided by the square of the variance of the
   * probability distribution minus 3)
   */
  virtual long double ExcessKurtosis() const
  {
    long double var = this->Variance();
    if(!std::isfinite(var))
      return NAN;
    long double mu = this->Mean(); /// var is finite, so is mu

    long double sum = this->ExpectedValue(
        [mu](double x) {
          double xmmu = x - mu;
          double kurtosisSqrt = xmmu * xmmu;
          double kurtosis = kurtosisSqrt * kurtosisSqrt;
          return kurtosis;
        },
        this->MinValue(), this->MaxValue());

    return sum / (var * var) - 3;
  }

  /**
   * @fn Kurtosis
   * @return unbiased kurtosis = μ_4 / σ ^ 4
   */
  virtual long double Kurtosis() const
  {
    return this->ExcessKurtosis() + 3.0;
  }

  /**
   * @fn SecondMoment
   * @return E[X^2]
   */
  virtual long double SecondMoment() const
  {
    long double mean = this->Mean();
    return mean * mean + this->Variance();
  }

  /**
   * @fn ThirdMoment
   * @return E[X^3]
   */
  virtual long double ThirdMoment() const
  {
    long double mean = this->Mean();
    long double variance = this->Variance();
    long double skewness = this->Skewness();

    long double moment = skewness * std::sqrt(variance) * variance;
    moment += mean * mean * mean;
    moment += 3 * mean * variance;
    return moment;
  }

  /**
   * @fn FourthMoment
   * @return E[X^4]
   */
  virtual long double FourthMoment() const
  {
    long double mean = this->Mean();
    long double variance = this->Variance();
    long double moment3 = this->ThirdMoment();
    long double kurtosis = this->Kurtosis();
    long double meanSq = mean * mean;

    long double moment = kurtosis * variance * variance;
    moment -= 6 * meanSq * variance;
    moment -= 3 * meanSq * meanSq;
    moment += 4 * mean * moment3;
    return moment;
  }

  //--------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn isLeftBounded
   * @return true if distribution is bounded from the left
   */
  bool isLeftBounded() const
  {
    SUPPORT_TYPE supp = this->SupportType();
    return (supp == RIGHTSEMIFINITE_T || supp == FINITE_T);
  }

  /**
   * @fn isRightBounded
   * @return true if distribution is bounded from the right
   */
  bool isRightBounded() const
  {
    SUPPORT_TYPE supp = this->SupportType();
    return (supp == LEFTSEMIFINITE_T || supp == FINITE_T);
  }

  /**
   * @fn Quantile
   * @param p
   * @return quantileImpl(p) if p is in (0, 1)
   */
  T Quantile(double p) const
  {
    if(p < 0.0 || p > 1.0)
      throw std::invalid_argument("Probability p in quantile function should be in interval [0, 1]");
    double minVal = this->MinValue();
    if(p == 0.0)
      return minVal;
    double maxVal = this->MaxValue();
    if(p == 1.0)
      return maxVal;
    double x = this->quantileImpl(p);
    if(x < minVal)
      return minVal;
    return (x > maxVal) ? maxVal : x;
  }

  /**
   * @fn Quantile1m
   * @param p
   * @return quantileImpl1m(p) if p is in (0, 1)
   */
  T Quantile1m(double p) const
  {
    if(p < 0.0 || p > 1.0)
      throw std::invalid_argument("Probability p in quantile function should be in interval [0, 1]");
    double minVal = this->MinValue();
    if(p == 1.0)
      return minVal;
    double maxVal = this->MaxValue();
    if(p == 0.0)
      return maxVal;
    double x = this->quantileImpl1m(p);
    if(x < minVal)
      return minVal;
    return (x > maxVal) ? maxVal : x;
  }

  /**
   * @fn QuantileFunction
   * @param p
   * @param y
   * @return fills vector y with Quantile(p)
   */
  void QuantileFunction(const std::vector<double>& p, std::vector<T>& y)
  {
    int size = std::min(p.size(), y.size());
    for(int i = 0; i != size; ++i)
      y[i] = this->Quantile(p[i]);
  }

  /**
   * @fn QuantileFunction1m
   * @param p
   * @param y
   * @return fills vector y with Quantile1m(p)
   */
  void QuantileFunction1m(const std::vector<double>& p, std::vector<T>& y)
  {
    int size = std::min(p.size(), y.size());
    for(int i = 0; i != size; ++i)
      y[i] = this->Quantile1m(p[i]);
  }

  /**
   * @fn CF
   * @param t
   * @return CFImpl for t != 0 and 1 otherwise
   */
  std::complex<double> CF(double t) const
  {
    if(t == 0.0)
      return 1.0;
    return (t > 0.0) ? this->CFImpl(t) : std::conj(this->CFImpl(-t));
  }

  /**
   * @fn CharacteristicFunction
   * @param x input vector
   * @param y output vector: y = CF(x)
   */
  void CharacteristicFunction(const std::vector<double>& t, std::vector<std::complex<double>>& y) const
  {
    size_t size = std::min(t.size(), y.size());
    for(size_t i = 0; i != size; ++i)
      y[i] = this->CF(t[i]);
  }

  /**
   * @fn HazardFunction
   * @param x input vector
   * @param y output vector: y = Hazard(x)
   */
  void HazardFunction(const std::vector<T>& x, std::vector<double>& y) const
  {
    int size = std::min(x.size(), y.size());
    for(int i = 0; i != size; ++i)
      y[i] = this->Hazard(x[i]);
  }

  /**
   * @fn GetSampleSum
   * @param sample
   * @return sum of all elements in a sample
   */
  static long double GetSampleSum(const std::vector<T>& sample)
  {
    return std::accumulate(sample.begin(), sample.end(), 0.0);
  }

  /**
   * @fn GetSampleMean
   * @param sample
   * @return arithmetic average
   */
  static long double GetSampleMean(const std::vector<T>& sample)
  {
    size_t n = sample.size();
    return (n > 0) ? GetSampleSum(sample) / n : 0.0;
  }

  /**
   * @fn GetSampleLogMean
   * @param sample
   * @return arithmetic log-average
   */
  static long double GetSampleLogMean(const std::vector<T>& sample)
  {
    long double sum = 0.0;
    for(const T& var : sample)
      sum += std::log(var);
    return sum / sample.size();
  }

  /**
   * @fn GetSampleVariance
   * @param sample
   * @param mean known mean value
   * @return sample second central moment
   */
  static long double GetSampleVariance(const std::vector<T>& sample, double mean)
  {
    long double sum = 0.0l;
    for(const T& var : sample)
    {
      double temp = var - mean;
      sum += temp * temp;
    }
    return sum / sample.size();
  }

  /**
   * @fn GetSampleLogVariance
   * @param sample
   * @param logMean known log-mean value
   * @return sample log-variance
   */
  static long double GetSampleLogVariance(const std::vector<T>& sample, double logMean)
  {
    long double sum = 0.0l;
    for(const T& var : sample)
    {
      double temp = std::log(var) - logMean;
      sum += temp * temp;
    }
    return sum / sample.size();
  }

  /**
   * @fn GetSampleMeanAndVariance
   * @param sample
   * @return sample mean and variance
   */
  static LongDoublePair GetSampleMeanAndVariance(const std::vector<T>& sample)
  {
    /// Welford's stable method
    long double m = 0.0l, v = 0.0l;
    int n = sample.size();
    for(int i = 0; i < n; ++i)
    {
      double x = sample[i];
      double diff = x - m;
      m += diff / (i + 1);
      v += diff * (x - m);
    }
    return std::make_pair(m, v / n);
  }

  /**
   * @fn GetSampleLogMeanAndVariance
   * @param sample
   * @return sample log-mean and log-variance
   */
  static LongDoublePair GetSampleLogMeanAndVariance(const std::vector<T>& sample)
  {
    /// Welford's stable method
    long double m = 0.0l, v = 0.0l;
    int n = sample.size();
    for(int i = 0; i < n; ++i)
    {
      double logX = std::log(sample[i]);
      double diff = logX - m;
      m += diff / (i + 1);
      v += diff * (logX - m);
    }
    return std::make_pair(m, v / n);
  }

  /**
   * @fn GetSampleStatistics
   * @param sample
   * @return sample mean, variance, skewness and excess kurtosis
   */
  static std::tuple<long double, long double, long double, long double> GetSampleStatistics(const std::vector<T>& sample)
  {
    /// Terriberry's extension for skewness and kurtosis
    long double M1{}, M2{}, M3{}, M4{};
    long double m1{}, m2{}, m3{}, m4{};
    size_t n = sample.size(), k = 0;
    size_t t = 0;
    static constexpr size_t BIG_NUMBER = 10000;
    for(const T& var : sample)
    {
      ++k;
      long double delta = var - m1;
      long double delta_k = delta / k;
      long double delta_kSq = delta_k * delta_k;
      long double term1 = delta * delta_k * (k - 1);
      m1 += delta_k;
      m4 += term1 * delta_kSq * (k * k - 3 * k + 3) + 6 * delta_kSq * m2 - 4 * delta_k * m3;
      m3 += term1 * delta_k * (k - 2) - 3 * delta_k * m2;
      m2 += term1;

      /// This looks like a hack and unfortunately it is. The reason of it is that
      /// the algorithm can become unstable for sufficiently large k. For now we
      /// restart the algorithm when k reaches some big number. In the future this
      /// can be parallelized for faster implementation.
      if(k >= BIG_NUMBER)
      {
        long double Delta = m1 - M1;
        long double DeltaSq = Delta * Delta;
        size_t tp1 = t + 1;
        /// M4
        M4 += m4 + (DeltaSq * DeltaSq * t * (t * t - t + 1) * BIG_NUMBER) / (tp1 * tp1 * tp1);
        M4 += 6 * DeltaSq * (t * t * m2 + M2) / (tp1 * tp1);
        M4 += 4 * Delta * (t * m3 - M3) / tp1;
        /// M3
        M3 += m3 + (DeltaSq * Delta * t * (t - 1) * BIG_NUMBER) / (tp1 * tp1);
        M3 += 3 * Delta * (t * m2 - M2) / tp1;
        /// M2 and M1
        M2 += m2 + (DeltaSq * t) / tp1;
        M1 += Delta / tp1;
        k = 0;
        m1 = 0;
        m2 = 0;
        m3 = 0;
        m4 = 0;
        ++t;
      }
    }

    /// If something left - add the residue
    long double res = static_cast<long double>(k) / BIG_NUMBER;
    if(res != 0)
    {
      long double Delta = m1 - M1;
      long double DeltaSq = Delta * Delta;
      long double tpres = t + res;
      /// M4
      M4 += m4 + (DeltaSq * DeltaSq * t * res * (t * t - res * t + res * res) * BIG_NUMBER) / (tpres * tpres * tpres);
      M4 += 6 * DeltaSq * (t * t * m2 + res * res * M2) / (tpres * tpres);
      M4 += 4 * Delta * (t * m3 - res * M3) / tpres;
      /// M3
      M3 += m3 + (DeltaSq * Delta * t * res * (t - res) * BIG_NUMBER) / (tpres * tpres);
      M3 += 3 * Delta * (t * m2 - res * M2) / tpres;
      /// M2 and M1
      M2 += m2 + (DeltaSq * t * res) / tpres;
      M1 += Delta * res / tpres;
    }

    long double variance = M2 / n;
    long double skewness = std::sqrt(n) * M3 / std::pow(M2, 1.5);
    long double exkurtosis = (n * M4) / (M2 * M2) - 3.0;
    return std::make_tuple(M1, variance, skewness, exkurtosis);
  }

protected:
  //--------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn ExpectedValue
   * @param funPtr pointer on function g(x) with finite support which expected
   * value should be returned
   * @param minPoint min{x | g(x) ≠ 0}
   * @param maxPoint max{x | g(x) ≠ 0}
   * @return E[g(x)]
   */
  virtual long double ExpectedValue(const std::function<double(T)>& funPtr, T minPoint, T maxPoint) const = 0;

  //--------------------------------------------------------------------------------------
  // VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn CFimpl
   * @param t
   * @return characteristic function implementation (inverse Fourier transform
   * of pdf)
   */
  virtual std::complex<double> CFImpl(double t) const
  {
    T leftBound = this->MinValue(), rightBound = this->MaxValue();
    if(leftBound == rightBound)
      return std::complex<double>(std::cos(t * leftBound), std::sin(t * leftBound));

    double re = this->ExpectedValue([t](double x) { return std::cos(t * x); }, leftBound, rightBound);

    double im = this->ExpectedValue([t](double x) { return std::sin(t * x); }, leftBound, rightBound);

    return {re, im};
  }

  //--------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //--------------------------------------------------------------------------------------

  /**
   * @fn allElementsAreNotGreaterThan
   * @param value
   * @param sample
   * @return true if all elements in sample are not greater than given value
   */
  static bool allElementsAreNotGreaterThan(T value, const std::vector<T>& sample)
  {
    for(const T& var : sample)
    {
      if(var > value)
        return false;
    }
    return true;
  }

  /**
   * @fn allElementsAreNotSmallerThan
   * @param value
   * @param sample
   * @return true if all elements in sample are not smaller than given value
   */
  static bool allElementsAreNotSmallerThan(T value, const std::vector<T>& sample)
  {
    for(const T& var : sample)
    {
      if(var < value)
        return false;
    }
    return true;
  }

  /**
   * @fn allElementsAreNonNegative
   * @param sample
   * @return true if all elements in sample are non-negative
   */
  static bool allElementsAreNonNegative(const std::vector<T>& sample)
  {
    return allElementsAreNotSmallerThan(0, sample);
  }

  /**
   * @fn allElementsArePositive
   * @param sample
   * @return true if all elements in sample are positive
   */
  static bool allElementsArePositive(const std::vector<T>& sample)
  {
    for(const T& var : sample)
    {
      if(var <= 0)
        return false;
    }
    return true;
  }

private:
  //--------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //--------------------------------------------------------------------------------------

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
};