#include "Distributions.h"

namespace distributions
{
template <typename T>
bool UnivariateDistribution<T>::isLeftBounded() const
{
  SUPPORT_TYPE supp = this->SupportType();
  return (supp == RIGHTSEMIFINITE_T || supp == FINITE_T);
}

template <typename T>
bool UnivariateDistribution<T>::isRightBounded() const
{
  SUPPORT_TYPE supp = this->SupportType();
  return (supp == LEFTSEMIFINITE_T || supp == FINITE_T);
}

template <typename T>
T UnivariateDistribution<T>::Quantile(double p) const
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

template <typename T>
T UnivariateDistribution<T>::Quantile1m(double p) const
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

template <typename T>
void UnivariateDistribution<T>::QuantileFunction(const std::vector<double>& p, std::vector<T>& y)
{
  int size = std::min(p.size(), y.size());
  for(int i = 0; i != size; ++i)
    y[i] = this->Quantile(p[i]);
}

template <typename T>
void UnivariateDistribution<T>::QuantileFunction1m(const std::vector<double>& p, std::vector<T>& y)
{
  int size = std::min(p.size(), y.size());
  for(int i = 0; i != size; ++i)
    y[i] = this->Quantile1m(p[i]);
}

template <typename T>
std::complex<double> UnivariateDistribution<T>::CF(double t) const
{
  if(t == 0.0)
    return 1.0;
  return (t > 0.0) ? this->CFImpl(t) : std::conj(this->CFImpl(-t));
}

template <typename T>
std::complex<double> UnivariateDistribution<T>::CFImpl(double t) const
{
  T leftBound = this->MinValue(), rightBound = this->MaxValue();
  if(leftBound == rightBound)
    return std::complex<double>(std::cos(t * leftBound), std::sin(t * leftBound));

  double re = this->ExpectedValue([this, t](double x) { return std::cos(t * x); }, leftBound, rightBound);

  double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, leftBound, rightBound);

  return std::complex<double>(re, im);
}

template <typename T>
void UnivariateDistribution<T>::CharacteristicFunction(const std::vector<double>& t, std::vector<std::complex<double>>& y) const
{
  int size = std::min(t.size(), y.size());
  for(int i = 0; i != size; ++i)
    y[i] = this->CF(t[i]);
}

template <typename T>
void UnivariateDistribution<T>::HazardFunction(const std::vector<T>& x, std::vector<double>& y) const
{
  int size = std::min(x.size(), y.size());
  for(int i = 0; i != size; ++i)
    y[i] = this->Hazard(x[i]);
}

template <typename T>
T UnivariateDistribution<T>::Median() const
{
  return quantileImpl(0.5);
}

template <typename T>
long double UnivariateDistribution<T>::Skewness() const
{
  long double var = this->Variance();
  if(!std::isfinite(var))
    return NAN;
  long double mu = this->Mean(); /// var is finite, so is mu

  long double sum = this->ExpectedValue(
      [this, mu](double x) {
        double xmmu = x - mu;
        double skewness = xmmu * xmmu * xmmu;
        return skewness;
      },
      this->MinValue(), this->MaxValue());

  return sum / std::pow(var, 1.5);
}

template <typename T>
long double UnivariateDistribution<T>::ExcessKurtosis() const
{
  long double var = this->Variance();
  if(!std::isfinite(var))
    return NAN;
  long double mu = this->Mean(); /// var is finite, so is mu

  long double sum = this->ExpectedValue(
      [this, mu](double x) {
        double xmmu = x - mu;
        double kurtosisSqrt = xmmu * xmmu;
        double kurtosis = kurtosisSqrt * kurtosisSqrt;
        return kurtosis;
      },
      this->MinValue(), this->MaxValue());

  return sum / (var * var) - 3;
}

template <typename T>
long double UnivariateDistribution<T>::Kurtosis() const
{
  return this->ExcessKurtosis() + 3.0;
}

template <typename T>
long double UnivariateDistribution<T>::SecondMoment() const
{
  long double mean = this->Mean();
  return mean * mean + this->Variance();
}

template <typename T>
long double UnivariateDistribution<T>::ThirdMoment() const
{
  long double mean = this->Mean();
  long double variance = this->Variance();
  long double skewness = this->Skewness();

  long double moment = skewness * std::sqrt(variance) * variance;
  moment += mean * mean * mean;
  moment += 3 * mean * variance;
  return moment;
}

template <typename T>
long double UnivariateDistribution<T>::FourthMoment() const
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

template <typename T>
bool UnivariateDistribution<T>::allElementsAreNotGreaterThan(T value, const std::vector<T>& sample)
{
  for(const T& var : sample)
  {
    if(var > value)
      return false;
  }
  return true;
}

template <typename T>
bool UnivariateDistribution<T>::allElementsAreNotSmallerThan(T value, const std::vector<T>& sample)
{
  for(const T& var : sample)
  {
    if(var < value)
      return false;
  }
  return true;
}

template <typename T>
bool UnivariateDistribution<T>::allElementsAreNonNegative(const std::vector<T>& sample)
{
  return allElementsAreNotSmallerThan(0, sample);
}

template <typename T>
bool UnivariateDistribution<T>::allElementsArePositive(const std::vector<T>& sample)
{
  for(const T& var : sample)
  {
    if(var <= 0)
      return false;
  }
  return true;
}

template <typename T>
long double UnivariateDistribution<T>::GetSampleSum(const std::vector<T>& sample)
{
  return std::accumulate(sample.begin(), sample.end(), 0.0);
}

template <typename T>
long double UnivariateDistribution<T>::GetSampleMean(const std::vector<T>& sample)
{
  size_t n = sample.size();
  return (n > 0) ? GetSampleSum(sample) / n : 0.0;
}

template <typename T>
long double UnivariateDistribution<T>::GetSampleLogMean(const std::vector<T>& sample)
{
  long double sum = 0.0;
  for(const T& var : sample)
    sum += std::log(var);
  return sum / sample.size();
}

template <typename T>
long double UnivariateDistribution<T>::GetSampleVariance(const std::vector<T>& sample, double mean)
{
  long double sum = 0.0l;
  for(const T& var : sample)
  {
    double temp = var - mean;
    sum += temp * temp;
  }
  return sum / sample.size();
}

template <typename T>
long double UnivariateDistribution<T>::GetSampleLogVariance(const std::vector<T>& sample, double logMean)
{
  long double sum = 0.0l;
  for(const T& var : sample)
  {
    double temp = std::log(var) - logMean;
    sum += temp * temp;
  }
  return sum / sample.size();
}

template <typename T>
LongDoublePair UnivariateDistribution<T>::GetSampleMeanAndVariance(const std::vector<T>& sample)
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

template <typename T>
LongDoublePair UnivariateDistribution<T>::GetSampleLogMeanAndVariance(const std::vector<T>& sample)
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

template <typename T>
std::tuple<long double, long double, long double, long double> UnivariateDistribution<T>::GetSampleStatistics(const std::vector<T>& sample)
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

    /// This looks like a hack and unfortunately it is. The reason of it is that the algorithm
    /// can become unstable for sufficiently large k. For now we restart the algorithm when
    /// k reaches some big number. In the future this can be parallelized for faster implementation.
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

template <class T1, class T2, typename T>
void BivariateDistribution<T1, T2, T>::Reseed(unsigned long seed) const
{
  this->localRandGenerator.Reseed(seed);
  X.Reseed(seed + 1);
  Y.Reseed(seed + 2);
}

template <class T1, class T2, typename T>
LongDoublePair BivariateDistribution<T1, T2, T>::Mean() const
{
  return std::make_pair(X.Mean(), Y.Mean());
}

template <class T1, class T2, typename T>
LongDoubleTriplet BivariateDistribution<T1, T2, T>::Covariance() const
{
  double var1 = X.Variance();
  double var2 = Y.Variance();
  double corr = Correlation() * var1 * var2;
  return std::make_tuple(var1, corr, var2);
}

template <class T1, class T2, typename T>
std::pair<T1, T2> BivariateDistribution<T1, T2, T>::GetMarginalDistributions() const
{
  return std::make_pair(X, Y);
}


//-------------------------------------------------------------------------------------------
// CONTINUOUS
//-------------------------------------------------------------------------------------------
template <typename RealType>
void ContinuousDistribution<RealType>::ProbabilityDensityFunction(const std::vector<RealType>& x, std::vector<double>& y) const
{
  for(size_t i = 0; i != x.size(); ++i)
    y[i] = this->f(x[i]);
}

template <typename RealType>
void ContinuousDistribution<RealType>::LogProbabilityDensityFunction(const std::vector<RealType>& x, std::vector<double>& y) const
{
  for(size_t i = 0; i != x.size(); ++i)
    y[i] = this->logf(x[i]);
}

template <typename RealType>
RealType ContinuousDistribution<RealType>::quantileImpl(double p, RealType initValue) const
{
  static constexpr double SMALL_P = 1e-5;
  if(p < SMALL_P)
  {
    /// for small p we use logarithmic scale
    double logP = std::log(p);
    if(!RandMath::findRootNewtonFirstOrder<RealType>(
           [this, logP](const RealType& x) {
             double logCdf = std::log(this->F(x)), logPdf = this->logf(x);
             double first = logCdf - logP;
             double second = std::exp(logPdf - logCdf);
             return DoublePair(first, second);
           },
           initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  if(this->SupportType() == FINITE_T)
  {
    if(!RandMath::findRootNewtonFirstOrder<RealType>([this, p](const RealType& x) { return this->F(x) - p; }, this->MinValue(), this->MaxValue(), initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  if(!RandMath::findRootNewtonFirstOrder<RealType>(
         [this, p](const RealType& x) {
           double first = this->F(x) - p;
           double second = this->f(x);
           return DoublePair(first, second);
         },
         initValue))
    throw std::runtime_error("Continuous distribution: failure in numeric procedure");
  return initValue;
}

template <typename RealType>
RealType ContinuousDistribution<RealType>::quantileImpl(double p) const
{
  /// We use quantile from sample as an initial guess
  static constexpr int SAMPLE_SIZE = 128;
  static std::vector<RealType> sample(SAMPLE_SIZE);
  this->Sample(sample);
  int index = p * SAMPLE_SIZE;
  if(index == 0.0)
    return this->quantileImpl(p, *std::min_element(sample.begin(), sample.end()));
  std::nth_element(sample.begin(), sample.begin() + index, sample.end());
  return this->quantileImpl(p, sample[index]);
}

template <typename RealType>
RealType ContinuousDistribution<RealType>::quantileImpl1m(double p, RealType initValue) const
{
  static constexpr double SMALL_P = 1e-5;
  if(p < SMALL_P)
  {
    /// for small p we use logarithmic scale
    double logP = std::log(p);
    if(!RandMath::findRootNewtonFirstOrder<RealType>(
           [this, logP](const RealType& x) {
             double logCcdf = std::log(this->S(x)), logPdf = this->logf(x);
             double first = logP - logCcdf;
             double second = std::exp(logPdf - logCcdf);
             return DoublePair(first, second);
           },
           initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  if(this->SupportType() == FINITE_T)
  {
    if(!RandMath::findRootNewtonFirstOrder<RealType>([this, p](const RealType& x) { return this->S(x) - p; }, this->MinValue(), this->MaxValue(), initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  if(!RandMath::findRootNewtonFirstOrder<RealType>(
         [this, p](const RealType& x) {
           double first = p - this->S(x);
           double second = this->f(x);
           return DoublePair(first, second);
         },
         initValue))
    throw std::runtime_error("Continuous distribution: failure in numeric procedure");
  return initValue;
}

template <typename RealType>
RealType ContinuousDistribution<RealType>::quantileImpl1m(double p) const
{
  /// We use quantile from sample as an initial guess
  static constexpr int SAMPLE_SIZE = 128;
  static std::vector<RealType> sample(SAMPLE_SIZE);
  this->Sample(sample);
  int index = p * SAMPLE_SIZE;
  if(index == 0.0)
    return this->quantileImpl1m(p, *std::max_element(sample.begin(), sample.end()));
  std::nth_element(sample.begin(), sample.begin() + index, sample.end(), std::greater<>());
  return this->quantileImpl1m(p, sample[index]);
}

template <typename RealType>
RealType ContinuousDistribution<RealType>::Mode() const
{
  RealType guess = this->Mean(); /// good starting point
  if(!std::isfinite(guess))
    guess = this->Median(); /// this shouldn't be nan or inf
  RealType root = 0;
  RandMath::findMin<RealType>([this](const RealType& x) { return -this->logf(x); }, guess, root);
  return root;
}

template <typename RealType>
long double ContinuousDistribution<RealType>::ExpectedValue(const std::function<double(RealType)>& funPtr, RealType minPoint, RealType maxPoint) const
{
  /// attempt to calculate expected value by numerical method
  /// use for distributions w/o explicit formula
  RealType lowerBoundary = minPoint, upperBoundary = maxPoint;
  if(this->isRightBounded())
    lowerBoundary = std::max(minPoint, lowerBoundary);
  if(this->isLeftBounded())
    upperBoundary = std::min(maxPoint, upperBoundary);

  if(lowerBoundary >= upperBoundary)
    return 0.0;

  bool isLeftBoundFinite = std::isfinite(lowerBoundary), isRightBoundFinite = std::isfinite(upperBoundary);

  /// Integrate on finite interval [a, b]
  if(isLeftBoundFinite && isRightBoundFinite)
  {
    return RandMath::integral(
        [this, funPtr](double x) {
          double y = funPtr(x);
          return (y == 0.0) ? 0.0 : y * f(x);
        },
        lowerBoundary, upperBoundary);
  }

  /// Integrate on semifinite interval [a, inf)
  if(isLeftBoundFinite)
  {
    return RandMath::integral(
        [this, funPtr, lowerBoundary](double x) {
          if(x >= 1.0)
            return 0.0;
          double denom = 1.0 - x;
          double t = lowerBoundary + x / denom;
          double y = funPtr(t);
          if(y == 0.0)
            return 0.0;
          y *= this->f(t);
          denom *= denom;
          return y / denom;
        },
        0.0, 1.0);
  }

  /// Integrate on semifinite intervale (-inf, b]
  if(isRightBoundFinite)
  {
    return RandMath::integral(
        [this, funPtr, upperBoundary](double x) {
          if(x <= 0.0)
            return 0.0;
          double t = upperBoundary - (1.0 - x) / x;
          double y = funPtr(t);
          if(y == 0.0)
            return 0.0;
          y *= this->f(t);
          return y / (x * x);
        },
        0.0, 1.0);
  }

  /// Infinite case
  return RandMath::integral(
      [this, funPtr](double x) {
        if(std::fabs(x) >= 1.0)
          return 0.0;
        double x2 = x * x;
        double denom = 1.0 - x2;
        double t = x / denom;
        double y = funPtr(t);
        if(y == 0.0)
          return 0.0;
        y *= this->f(t);
        denom *= denom;
        return y * (1.0 + x2) / denom;
      },
      -1.0, 1.0);
}

template <typename RealType>
double ContinuousDistribution<RealType>::Hazard(const RealType& x) const
{
  if(x < this->MinValue())
    return 0.0; /// 0/1
  if(x > this->MaxValue())
    return NAN; /// 0/0
  return this->f(x) / this->S(x);
}

template <typename RealType>
double ContinuousDistribution<RealType>::LikelihoodFunction(const std::vector<RealType>& sample) const
{
  return std::exp(LogLikelihoodFunction(sample));
}

template <typename RealType>
double ContinuousDistribution<RealType>::LogLikelihoodFunction(const std::vector<RealType>& sample) const
{
  long double res = 0.0;
  for(const RealType& var : sample)
    res += this->logf(var);
  return res;
}

template <typename RealType>
bool ContinuousDistribution<RealType>::KolmogorovSmirnovTest(const std::vector<RealType>& orderStatistic, double alpha) const
{
  KolmogorovSmirnovRand KSRand;
  double K = KSRand.Quantile1m(alpha);
  size_t n = orderStatistic.size();
  double interval = K / std::sqrt(n);
  double nInv = 1.0 / n;
  double Fn = 0.0;
  for(size_t i = 1; i != n; ++i)
  {
    RealType x = orderStatistic[i - 1];
    if(x > orderStatistic[i])
      throw std::invalid_argument("Order statistic should be sorted in ascending order");
    double upperBound = Fn + interval;
    Fn = i * nInv;
    double lowerBound = Fn - interval;
    double FReal = this->F(x);
    if(FReal < lowerBound || FReal > upperBound)
      return false;
  }
  double SReal = this->S(orderStatistic[n - 1]);
  return (SReal > interval || SReal < nInv - interval) ? false : true;
}

template < typename RealType >
CircularDistribution<RealType>::CircularDistribution(double location)
{
  SetLocation(location);
}

template < typename RealType >
void CircularDistribution<RealType>::SetLocation(double location)
{
  loc = location;
}


//-------------------------------------------------------------------------------------------
// DISCRETE
//-------------------------------------------------------------------------------------------
template < typename IntType >
void DiscreteDistribution<IntType>::ProbabilityMassFunction(const std::vector<IntType> &x, std::vector<double> &y) const
{
  for (size_t i = 0; i != x.size(); ++i)
    y[i] = this->P(x[i]);
}

template < typename IntType >
void DiscreteDistribution<IntType>::LogProbabilityMassFunction(const std::vector<IntType> &x, std::vector<double> &y) const
{
  for (size_t i = 0; i != x.size(); ++i)
    y[i] = this->logP(x[i]);
}

template < typename IntType >
IntType DiscreteDistribution<IntType>::Mode() const
{
  /// Works only for unimodal distributions
  IntType x = this->Median();
  double logProb = this->logP(x), newLogProb = this->logP(x + 1);
  if (logProb < newLogProb) {
    do {
      ++x;
      logProb = newLogProb;
      newLogProb = this->logP(x + 1);
    } while (logProb < newLogProb);
  }
  else {
    newLogProb = this->logP(x - 1);
    while (logProb < newLogProb) {
      --x;
      logProb = newLogProb;
      newLogProb = this->logP(x - 1);
    }
  }
  return x;
}

template < typename IntType >
IntType DiscreteDistribution<IntType>::quantileImpl(double p, IntType initValue) const
{
  IntType down = initValue, up = down + 1;
  double fu = this->F(up), fd = this->F(down);
  /// go up
  while (fu < p) {
    fd = fu;
    fu = this->F(++up);
  }
  down = up - 1;
  /// go down
  while (fd > p) {
    fd = this->F(--down);
  }
  up = down + 1;
  /// if lower quantile is not equal probability, we return upper quantile
  return (fd < p) ? up : down;
}


template < typename IntType >
IntType DiscreteDistribution<IntType>::quantileImpl(double p) const
{
  /// We use quantile from sample as an initial guess
  static constexpr int SAMPLE_SIZE = 128;
  static std::vector<IntType> sample(SAMPLE_SIZE);
  this->Sample(sample);
  int index = p * SAMPLE_SIZE;
  if (index == 0)
    return this->quantileImpl(p, *std::min_element(sample.begin(), sample.end()));
  std::nth_element(sample.begin(), sample.begin() + index, sample.end());
  return this->quantileImpl(p, sample[index]);
}

template < typename IntType >
IntType DiscreteDistribution<IntType>::quantileImpl1m(double p, IntType initValue) const
{
  IntType down = initValue, up = down + 1;
  double su = this->S(up), sd = this->S(down);
  /// go up
  while (su > p) {
    sd = su;
    su = this->S(++up);
  }
  down = up - 1;
  /// go down
  while (sd < p) {
    sd = this->S(--down);
  }
  up = down + 1;

  /// if lower quantile is not equal probability, we return upper quantile
  return (sd > p) ? up : down;
}

template < typename IntType >
IntType DiscreteDistribution<IntType>::quantileImpl1m(double p) const
{
  /// We use quantile from sample as an initial guess
  static constexpr int SAMPLE_SIZE = 128;
  static std::vector<IntType> sample(SAMPLE_SIZE);
  this->Sample(sample);
  int index = p * SAMPLE_SIZE;
  if (index == 0)
    return this->quantileImpl1m(p, *std::max_element(sample.begin(), sample.end()));
  std::nth_element(sample.begin(), sample.begin() + index, sample.end(), std::greater<>());
  return this->quantileImpl1m(p, sample[index]);
}

template < typename IntType >
long double DiscreteDistribution<IntType>::ExpectedValue(const std::function<double (IntType)> &funPtr, IntType minPoint, IntType maxPoint) const
{
  SUPPORT_TYPE suppType = this->SupportType();
  IntType k = minPoint, upperBoundary = maxPoint;
  if (suppType == FINITE_T || suppType == RIGHTSEMIFINITE_T) {
    k = std::max(k, this->MinValue());
  }
  if (suppType == FINITE_T || suppType == LEFTSEMIFINITE_T) {
    upperBoundary = std::min(upperBoundary, this->MaxValue());
  }

  double sum = 0;
  do {
    double addon = funPtr(k);
    if (addon != 0.0) {
      double prob = this->P(k);
      if (prob < MIN_POSITIVE)
        return sum;
      addon *= this->P(k);
      sum += addon;
    }
    ++k;
  } while (k <= upperBoundary);
  return sum;
}

template < typename IntType >
double DiscreteDistribution<IntType>::Hazard(const IntType &x) const
{
  if (x < this->MinValue())
    return 0.0; /// 0/1
  if (x > this->MaxValue())
    return NAN; /// 0/0
  return this->P(x) / this->S(x);
}

template < typename IntType >
double DiscreteDistribution<IntType>::LikelihoodFunction(const std::vector<IntType> &sample) const
{
  long double res = 1.0;
  for (const IntType & var : sample )
    res *= this->P(var);
  return res;
}

template < typename IntType >
double DiscreteDistribution<IntType>::LogLikelihoodFunction(const std::vector<IntType> &sample) const
{
  long double res = 0.0;
  for (const IntType & var : sample )
    res += this->logP(var);
  return res;
}

template < typename IntType >
bool DiscreteDistribution<IntType>::PearsonChiSquaredTest(const std::vector<IntType> &orderStatistic, double alpha, int lowerBoundary, int upperBoundary, size_t numberOfEstimatedParameters) const
{
  size_t n = orderStatistic.size(), i = 0, k = 0;
  double nInv = 1.0 / n, sum = 0.0;

  /// Sanity checks
  if (lowerBoundary >= upperBoundary)
    throw std::invalid_argument("Lower boundary should be smaller than upper one");
  for (size_t j = 1; j != n; ++j) {
    if (orderStatistic[i] < orderStatistic[j - 1])
      throw std::invalid_argument("Order statistic should be sorted in ascending order");
  }
  if (orderStatistic[0] < this->MinValue())
    throw std::invalid_argument("Some elements in the sample are too small to belong to this distribution, they should be greater than "
                                + this->toStringWithPrecision(this->MinValue()));
  if (orderStatistic[n - 1] > this->MaxValue())
    throw std::invalid_argument("Some elements in the sample are too large to belong to this distribution, they should be smaller than "
                                + this->toStringWithPrecision(this->MaxValue()));

  /// Lower interval
  IntType x = orderStatistic[0];
  if (lowerBoundary > this->MinValue()) {
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
  while (i < n && x < upperBoundary) {
    size_t count = 1;
    x = orderStatistic[i];
    while (i + count < n && x == orderStatistic[i + count])
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
  if (upperBoundary < this->MaxValue()) {
    double prob = nInv * (n - i), expectedProb = this->S(upperBoundary);
    double addon = prob - expectedProb;
    addon *= addon;
    addon /= expectedProb;
    sum += addon;
    ++k;
  }

  if (k <= numberOfEstimatedParameters + 1) {
    throw std::invalid_argument("Sample is too small, number of groups (" + this->toStringWithPrecision(k)
                                + ") should be bigger than number of estimated parameters plus 1 ("
                                + this->toStringWithPrecision(numberOfEstimatedParameters + 1) + ")");
  }
  double statistic = n * sum;
  ChiSquaredRand X(k - 1);
  double q = X.Quantile1m(alpha);
  return statistic <= q;
}

template < typename IntType >
bool DiscreteDistribution<IntType>::PearsonChiSquaredTest(const std::vector<IntType> &orderStatistic, double alpha, size_t numberOfEstimatedParameters) const
{
  return PearsonChiSquaredTest(orderStatistic, alpha, this->MinValue(), this->MaxValue(), numberOfEstimatedParameters);
}


//-------------------------------------------------------------------------------------------
// SINGULAR
//-------------------------------------------------------------------------------------------
double SingularDistribution::Hazard(const double &) const
{
  return NAN;
}

long double SingularDistribution::ExpectedValue(const std::function<double (double)> &, double, double ) const
{
  return NAN;
}

double SingularDistribution::Mode() const
{
  return NAN;
}

double SingularDistribution::LikelihoodFunction(const std::vector<double> &) const
{
  return NAN;
}

double SingularDistribution::LogLikelihoodFunction(const std::vector<double> &) const
{
  return NAN;
}
} // namespace distributions