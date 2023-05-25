#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"

namespace randlib
{
/**
 * @brief The ExponentiallyModifiedGaussianRand class <BR>
 * Exponentially-modified Gaussian distribution
 *
 * Notation: X ~ EMG(μ, σ, β)
 *
 * Related distributions: <BR>
 * X = Y + Z, where Y ~ Normal(μ, σ) and Z ~ Exp(β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ExponentiallyModifiedGaussianRand : public randlib::ContinuousDistribution<RealType>
{
  NormalRand<RealType> X{};
  ExponentialRand<RealType> Y{};

  double a = 1;         ///< μ + βσ^2
  double b = M_SQRT1_2; ///< 1 / (√2 * σ)
  double c = 0.5;       ///< μ + βσ^2 / 2
  double v = 1;         /// βσ

public:
  explicit ExponentiallyModifiedGaussianRand(double location = 0, double variance = 1, double rate = 1)
  {
    SetParameters(location, variance, rate);
  }

  String Name() const override
  {
    return "Exponentially modified Gaussian(" + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(X.Variance()) + ", " + this->toStringWithPrecision(GetRate()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::INFINITE_T;
  }

  RealType MinValue() const override
  {
    return -INFINITY;
  }

  RealType MaxValue() const override
  {
    return INFINITY;
  }

  void SetParameters(double location, double variance, double rate)
  {
    if(variance <= 0)
      throw std::invalid_argument("Exponentially modified Gaussian distribution: "
                                  "variance should be positive");
    if(rate <= 0)
      throw std::invalid_argument("Exponentially modified Gaussian distribution: "
                                  "rate should be positive");

    X.SetLocation(location);
    X.SetVariance(variance);
    Y.SetRate(rate);

    double mu = X.GetLocation();
    double sigma = X.GetScale();
    double beta = Y.GetRate();
    double var = sigma * sigma;
    a = 0.5 * beta * var;
    c = mu + a;
    a += c;
    b = M_SQRT1_2 / sigma;
    v = beta * sigma;
  }

  inline double GetLocation() const
  {
    return X.GetLocation();
  }

  inline double GetScale() const
  {
    return X.GetScale();
  }

  inline double GetRate() const
  {
    return Y.GetRate();
  }

  double f(const RealType& x) const override
  {
    auto [y, exponent] = faux(x);
    return y * std::exp(exponent);
  }

  double logf(const RealType& x) const override
  {
    auto [y, exponent] = faux(x);
    return std::log(y) + exponent;
  }

  double F(const RealType& x) const override
  {
    double u = Y.GetRate() * (x - X.GetLocation());
    double y = X.F(x);
    double exponent = -u + 0.5 * v * v;
    exponent = std::exp(exponent);
    exponent *= X.F(x - v * X.GetScale());
    return y - exponent;
  }

  double S(const RealType& x) const override
  {
    double u = Y.GetRate() * (x - X.GetLocation());
    double y = X.S(x);
    double exponent = -u + 0.5 * v * v;
    exponent = std::exp(exponent);
    exponent *= X.F(x - v * X.GetScale());
    return y + exponent;
  }

  RealType Variate() const override
  {
    return X.Variate() + Y.Variate();
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    return NormalRand<RealType>::StandardVariate(randGenerator) + ExponentialRand<RealType>::StandardVariate(randGenerator);
  }

  void Reseed(unsigned long seed) const override
  {
    X.Reseed(seed);
    Y.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return X.Mean() + Y.Mean();
  }

  long double Variance() const override
  {
    return X.Variance() + Y.Variance();
  }

  long double Skewness() const override
  {
    long double sigma = X.GetScale();
    long double lambda = Y.GetRate();
    long double tmp = 1.0 / (sigma * lambda);
    long double tmpSq = tmp * tmp;
    long double y = 1.0 + tmpSq;
    y = y * y * y;
    y = std::sqrt(y);
    y = tmpSq * tmp / y;
    return y + y;
  }

  long double ExcessKurtosis() const override
  {
    long double sigma = X.GetScale();
    long double lambda = Y.GetRate();
    long double tmp = 1.0 / (sigma * lambda);
    tmp *= tmp;
    long double numerator = 1.0 + 2.0 * tmp + 3.0 * tmp * tmp;
    long double denominator = 1.0 + tmp;
    denominator *= denominator;
    long double y = numerator / denominator - 1.0;
    return 3.0 * y;
  }

private:
  DoublePair faux(const RealType& x) const
  {
    double lambda = Y.GetRate();
    double y = a - x;
    y *= b;
    y = std::erfc(y);
    y *= 0.5 * lambda;
    double exponent = c - x;
    exponent *= lambda;
    return std::make_pair(y, exponent);
  }

  std::complex<double> CFImpl(double t) const override
  {
    return X.CF(t) * Y.CF(t);
  }
};
} // namespace randlib
