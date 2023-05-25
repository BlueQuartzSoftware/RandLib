#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"

#include "external/pow.hpp"
#include "external/sqrt.hpp"

namespace randlib
{
/**
 * @brief The NakagamiDistribution class <BR>
 * Abstract class for Nakagami distribution
 *
 * Notation: X ~ Nakagami(μ, ω)
 *
 * Related distributions: <BR>
 * σX ~ Nakagami(μ, ωσ^2) <BR>
 * X^2 ~ Γ(μ, μ / ω)
 */
template <typename RealType = double>
class RANDLIB_EXPORT NakagamiDistribution : public randlib::ContinuousDistribution<RealType>
{
  double mu = 0.5;  ///< shape μ
  double omega = 1; ///< spread ω
  randlib::GammaRand<RealType> Y{};
  double lgammaShapeRatio = 0; ///< log(Γ(μ + 0.5) / Γ(μ))

protected:
  NakagamiDistribution(double shape = 0.5, double spread = 1)
  {
    SetParameters(shape, spread);
  }

public:
  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  RealType MinValue() const override
  {
    return 0;
  }

  RealType MaxValue() const override
  {
    return INFINITY;
  }

  /**
   * @fn GetShape
   * @return shape μ
   */
  inline double GetShape() const
  {
    return mu;
  }

  /**
   * @fn GetSpread
   * @return spread w
   */
  inline double GetSpread() const
  {
    return omega;
  }

  /**
   * @fn GetLogGammaFunction
   * @return log(Γ(μ))
   */
  inline double GetLogGammaFunction() const
  {
    return Y.GetLogGammaShape();
  }

  /**
   * @fn GetLogGammaShapeRatio
   * @return log(Γ(μ + 0.5) / Γ(μ))
   */
  inline double GetLogGammaShapeRatio() const
  {
    return lgammaShapeRatio;
  }

  double f(const RealType& x) const override
  {
    if(x < 0.0)
      return 0.0;
    if(x == 0)
    {
      if(mu > 0.5)
        return 0.0;
      return (mu < 0.5) ? INFINITY : std::sqrt(M_2_PI / omega);
    }
    return 2 * x * Y.f(x * x);
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0)
    {
      if(mu > 0.5)
        return -INFINITY;
      return (mu < 0.5) ? INFINITY : 0.5 * (M_LN2 - M_LNPI - std::log(omega));
    }
    return M_LN2 + std::log(x) + Y.logf(x * x);
  }

  double F(const RealType& x) const override
  {
    return (x > 0.0) ? Y.F(x * x) : 0.0;
  }

  double S(const RealType& x) const override
  {
    return (x > 0.0) ? Y.S(x * x) : 1.0;
  }

  RealType Variate() const override
  {
    return std::sqrt(Y.Variate());
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    Y.Sample(outputData);
    for(RealType& var : outputData)
      var = std::sqrt(var);
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    Y.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    long double y = lgammaShapeRatio;
    y -= 0.5 * Y.GetLogRate();
    return std::exp(y);
  }

  long double Variance() const override
  {
    long double y = lgammaShapeRatio;
    y = std::exp(2 * y);
    return omega * (1 - y / mu);
  }

  RealType Median() const override
  {
    return std::sqrt(Y.Quantile(0.5));
  }

  RealType Mode() const override
  {
    long double mode = 1.0 - 0.5 / mu;
    if(mode <= 0.0)
      return 0.0;
    return std::sqrt(omega * mode);
  }

  long double Skewness() const override
  {
    long double thirdMoment = lgammaShapeRatio;
    thirdMoment -= 1.5 * Y.GetLogRate();
    thirdMoment = (mu + 0.5) * std::exp(thirdMoment);
    long double mean = Mean();
    long double variance = Variance();
    return (thirdMoment - mean * (3 * variance + mean * mean)) / std::pow(variance, 1.5);
  }

  long double FourthMoment() const override
  {
    long double fourthMoment = omega / mu;
    fourthMoment *= fourthMoment;
    fourthMoment *= mu * (mu + 1);
    return fourthMoment;
  }

  long double ExcessKurtosis() const override
  {
    long double mean = Mean();
    long double secondMoment = this->SecondMoment();
    long double thirdMoment = this->ThirdMoment();
    long double fourthMoment = FourthMoment();
    long double meanSq = mean * mean;
    long double variance = secondMoment - meanSq;
    long double numerator = fourthMoment - 4 * thirdMoment * mean + 6 * secondMoment * meanSq - 3 * meanSq * meanSq;
    long double denominator = variance * variance;
    return numerator / denominator - 3.0;
  }

protected:
  /**
   * @fn SetParameters
   * @param shape μ
   * @param spread ω
   */
  void SetParameters(double shape, double spread)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Nakagami distribution: shape should be positive");
    if(spread <= 0.0)
      throw std::invalid_argument("Nakagami distribution: spread should be positive");
    mu = shape;
    omega = spread;
    Y.SetParameters(mu, mu / omega);
    lgammaShapeRatio = std::lgammal(mu + 0.5) - Y.GetLogGammaShape();
  }

  RealType quantileImpl(double p) const override
  {
    return std::sqrt(Y.Quantile(p));
  }

  RealType quantileImpl1m(double p) const override
  {
    return std::sqrt(Y.Quantile1m(p));
  }

  std::complex<double> CFImpl(double t) const override
  {
    if(mu >= 0.5)
      return randlib::ContinuousDistribution<RealType>::CFImpl(t);

    double re = this->ExpectedValue(
                    [this, t](double x) {
                      if(x == 0.0)
                        return 0.0;
                      return std::cos(t * x) - 1.0;
                    },
                    0, INFINITY) +
                1.0;

    double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, 0, INFINITY);

    return std::complex<double>(re, im);
  }
};

/**
 * @brief The NakagamiRand class <BR>
 * Nakagami distribution
 */
template <typename RealType = double>
class RANDLIB_EXPORT NakagamiRand : public NakagamiDistribution<RealType>
{
public:
  NakagamiRand(double shape = 0.5, double spread = 1)
  : NakagamiDistribution<RealType>(shape, spread)
  {
  }

  String Name() const override
  {
    return "Nakagami(" + this->toStringWithPrecision(this->GetShape()) + ", " + this->toStringWithPrecision(this->GetSpread()) + ")";
  }

  using NakagamiDistribution<RealType>::SetParameters;
};

/**
 * @brief The ChiRand class <BR>
 * Chi distribution
 *
 * Notation: X ~ χ(k)
 *
 * Related distributions: <BR>
 * X ~ Nakagami(k/2, k) <BR>
 * X^2 ~ χ^2(k) <BR>
 * X^2 ~ Γ(k/2, 0.5)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ChiRand : public NakagamiDistribution<RealType>
{

public:
  explicit ChiRand(int degree)
  {
    SetDegree(degree);
  }

  String Name() const override
  {
    return "Chi(" + this->toStringWithPrecision(GetDegree()) + ")";
  }

  /**
   * @fn SetDegree
   * set degree k
   * @param degree
   */
  void SetDegree(int degree)
  {
    if(degree < 1)
      throw std::invalid_argument("Chi distribution: degree parameter should be positive");
    NakagamiDistribution<RealType>::SetParameters(0.5 * degree, degree);
  }

  /**
   * @fn GetDegree
   * @return degree k
   */
  inline int GetDegree() const
  {
    return 2 * NakagamiDistribution<RealType>::GetShape();
  }

  long double Skewness() const override
  {
    long double mean = this->Mean();
    long double sigmaSq = this->Variance();
    long double skew = mean * (1 - 2 * sigmaSq);
    skew /= std::pow(sigmaSq, 1.5);
    return skew;
  }

  long double ExcessKurtosis() const override
  {
    long double mean = this->Mean();
    long double sigmaSq = this->Variance();
    long double sigma = std::sqrt(sigmaSq);
    long double skew = Skewness();
    long double kurt = 1.0 - mean * sigma * skew;
    kurt /= sigmaSq;
    --kurt;
    return 2 * kurt;
  }
};

/**
 * @brief The MaxwellBoltzmannRand class <BR>
 * Maxwell-Boltzmann distribution
 *
 * Notation: X ~ MB(σ)
 *
 * Related distributions: <BR>
 * X / σ ~ χ(3) <BR>
 * X ~ Nakagami(1.5, 3σ^2)
 */
template <typename RealType = double>
class RANDLIB_EXPORT MaxwellBoltzmannRand : public NakagamiDistribution<RealType>
{
  double sigma = 1; ///< scale σ
public:
  explicit MaxwellBoltzmannRand(double scale)
  {
    SetScale(scale);
  }

  String Name() const override
  {
    return "Maxwell-Boltzmann(" + this->toStringWithPrecision(GetScale()) + ")";
  }

  /**
   * @fn SetScale
   * set scale σ
   * @param scale
   */
  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Maxwell-Boltzmann distribution: scale should be positive");
    sigma = scale;
    NakagamiDistribution<RealType>::SetParameters(1.5, 3 * sigma * sigma);
  }

  /**
   * @fn GetScale
   * @return scale σ
   */
  double GetScale() const
  {
    return sigma;
  }

  double f(const RealType& x) const override
  {
    if(x <= 0)
      return 0;
    double xAdj = x / sigma;
    double xAdjSq = xAdj * xAdj;
    double y = std::exp(-0.5 * xAdjSq);
    return M_SQRT2 * M_1_SQRTPI * xAdjSq * y / sigma;
  }

  double F(const RealType& x) const override
  {
    if(x <= 0.0)
      return 0.0;
    double xAdj = M_SQRT1_2 * x / sigma;
    double y = std::exp(-xAdj * xAdj);
    y *= 2 * xAdj * M_1_SQRTPI;
    return std::erf(xAdj) - y;
  }

  double S(const RealType& x) const override
  {
    if(x <= 0.0)
      return 1.0;
    double xAdj = M_SQRT1_2 * x / sigma;
    double y = std::exp(-xAdj * xAdj);
    y *= 2 * xAdj * M_1_SQRTPI;
    return std::erfc(xAdj) + y;
  }

  RealType Variate() const override
  {
    RealType W = randlib::ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType N = randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    return sigma * std::sqrt(2 * W + N * N);
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    for(RealType& var : outputData)
      var = this->Variate();
  }

  long double Mean() const override
  {
    return 2 * M_1_SQRTPI * M_SQRT2 * sigma;
  }

  long double Variance() const override
  {
    return (3 - 8.0 * M_1_PI) * sigma * sigma;
  }

  RealType Mode() const override
  {
    return M_SQRT2 * sigma;
  }

  long double Skewness() const override
  {
    long double skewness = 3 * M_PI - 8;
    skewness = 2.0 / skewness;
    skewness *= std::sqrt(skewness);
    return (16 - 5 * M_PI) * skewness;
  }

  long double ExcessKurtosis() const override
  {
    long double numerator = 40 - 3 * M_PI;
    numerator *= M_PI;
    numerator -= 96;
    long double denominator = 3 * M_PI - 8;
    denominator *= denominator;
    return 4 * numerator / denominator;
  }
};

/**
 * @brief The RayleighRand class <BR>
 * Rayleigh distribution
 *
 * Notation: X ~ Rayleigh(σ)
 *
 * Related distributions:
 * X / σ ~ χ(2)
 * X ~ Nakagami(1, 2σ^2)
 */
template <typename RealType = double>
class RANDLIB_EXPORT RayleighRand : public NakagamiDistribution<RealType>
{
  double sigma = 1; ///< scale σ
public:
  explicit RayleighRand(double scale = 1)
  {
    SetScale(scale);
  }

  String Name() const override
  {
    return "Rayleigh(" + this->toStringWithPrecision(GetScale()) + ")";
  }

  /**
   * @fn SetScale
   * set scale σ
   * @param scale
   */
  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Rayleigh distribution: scale should be positive");
    sigma = scale;
    NakagamiDistribution<RealType>::SetParameters(1, 2 * sigma * sigma);
  }

  /**
   * @fn GetScale
   * @return scale σ
   */
  double GetScale() const
  {
    return sigma;
  }

  double f(const RealType& x) const override
  {
    if(x <= 0)
      return 0.0;
    double y = x / (sigma * sigma);
    return y * std::exp(-0.5 * x * y);
  }

  double F(const RealType& x) const override
  {
    if(x <= 0)
      return 0.0;
    double xAdj = x / sigma;
    return -std::expm1l(-0.5 * xAdj * xAdj);
  }

  double S(const RealType& x) const override
  {
    if(x <= 0)
      return 1.0;
    double xAdj = x / sigma;
    return std::exp(-0.5 * xAdj * xAdj);
  }

  RealType Variate() const override
  {
    double W = randlib::ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    return sigma * std::sqrt(2 * W);
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    for(RealType& var : outputData)
      var = this->Variate();
  }

  long double Mean() const override
  {
    return sigma * M_SQRTPI * M_SQRT1_2;
  }

  long double Variance() const override
  {
    return (2.0 - M_PI_2) * sigma * sigma;
  }

  RealType Median() const override
  {
    static constexpr double medianCoef = nonstd::sqrt(M_LN2 + M_LN2);
    return sigma * medianCoef;
  }

  RealType Mode() const override
  {
    return sigma;
  }

  long double Skewness() const override
  {
    static constexpr long double skewness = 2 * M_SQRTPI * (M_PI - 3) / nonstd::pow(4.0 - M_PI, 1.5);
    return skewness;
  }

  long double ExcessKurtosis() const override
  {
    static constexpr long double temp = 4 - M_PI;
    static constexpr long double kurtosis = -(6 * M_PI * M_PI - 24 * M_PI + 16) / (temp * temp);
    return kurtosis;
  }

  /**
   * @fn Fit
   * fit scale via maximum-likelihood method if unbiased = false,
   * otherwise set scale, returned by uniformly minimum variance unbiased
   * estimator
   * @param sample
   */
  void Fit(const std::vector<RealType>& sample, bool unbiased = false)
  {
    /// Sanity check
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    size_t n = sample.size();
    DoublePair stats = this->GetSampleMeanAndVariance(sample);
    double rawMoment = stats.second + stats.first * stats.first;
    double sigmaBiasedSq = 0.5 * rawMoment;
    if(!unbiased)
    {
      SetScale(std::sqrt(sigmaBiasedSq));
    }
    /// Calculate unbiased sigma
    else if(n > 100)
    {
      double coef = 1.0 / (640 * std::pow(n, 5));
      coef -= 1.0 / (192 * std::pow(n, 3));
      coef += 0.125 / n;
      SetScale(std::exp(std::log1pl(coef) + 0.5 * sigmaBiasedSq)); /// err ~ o(n^{-6.5}) < 1e-13
    }
    else if(n > 15)
    {
      double coef = RandMath::lfact(n);
      coef += RandMath::lfact(n - 1);
      coef += 2 * n * M_LN2;
      coef += 0.5 * std::log(n);
      coef -= RandMath::lfact(2 * n);
      coef -= 0.5 * M_LNPI;
      coef += 0.5 * std::log(sigmaBiasedSq);
      SetScale(std::exp(coef));
    }
    else
    {
      double scale = RandMath::lfact(n - 1);
      scale += RandMath::lfact(n);
      scale += 0.5 * std::log(n / M_PI * sigmaBiasedSq);
      scale += 2 * n * M_LN2;
      scale -= RandMath::lfact(2 * n);
      SetScale(scale);
    }
  }

private:
  RealType quantileImpl(double p) const override
  {
    return sigma * std::sqrt(-2 * std::log1pl(-p));
  }

  RealType quantileImpl1m(double p) const override
  {
    return sigma * std::sqrt(-2 * std::log(p));
  }
};
} // namespace randlib
