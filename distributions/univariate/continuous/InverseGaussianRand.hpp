#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/NormalRand.hpp"

namespace randlib
{
/**
 * @brief The InverseGaussianRand class <BR>
 * Inverse Gaussian (Wald) distribution
 *
 * Notation: X ~ IG(μ, λ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT InverseGaussianRand : public randlib::ContinuousDistribution<RealType>
{
  double mu = 1;                         ///< mean μ
  double lambda = 1;                     ///< shape λ
  double pdfCoef = M_SQRT1_2 / M_SQRTPI; ///< (λ/(2π))^(1/2)
  double cdfCoef = M_E * M_E;            ///< exp(2λ/μ)
public:
  InverseGaussianRand(double mean = 1, double shape = 1)
  {
    SetParameters(mean, shape);
  }

  String Name() const override
  {
    return "Inverse-Gaussian(" + this->toStringWithPrecision(GetMean()) + ", " + this->toStringWithPrecision(GetShape()) + ")";
  }

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

  void SetParameters(double mean, double shape)
  {
    if(mean <= 0.0)
      throw std::invalid_argument("Inverse-Gaussian distribution: mean should be positive");
    if(shape <= 0.0)
      throw std::invalid_argument("Inverse-Gaussian distribution: shape should be positive");
    mu = mean;
    lambda = shape;

    pdfCoef = 0.5 * std::log(0.5 * lambda * M_1_PI);
    cdfCoef = std::exp(2 * lambda / mu);
  }

  inline double GetMean() const
  {
    return mu;
  }

  inline double GetShape() const
  {
    return lambda;
  }

  double f(const RealType& x) const override
  {
    return (x > 0.0) ? std::exp(logf(x)) : 0.0;
  }

  double logf(const RealType& x) const override
  {
    if(x <= 0.0)
      return -INFINITY;
    double y = -1.5 * std::log(x);
    double z = (x - mu);
    z *= z;
    z *= -0.5 * lambda / (x * mu * mu);
    z += pdfCoef;
    return y + z;
  }

  double F(const RealType& x) const override
  {
    if(x <= 0.0)
      return 0.0;
    double b = std::sqrt(0.5 * lambda / x);
    double a = b * x / mu;
    double y = std::erfc(b - a);
    y += cdfCoef * std::erfc(a + b);
    return 0.5 * y;
  }

  double S(const RealType& x) const override
  {
    if(x <= 0.0)
      return 1.0;
    double b = std::sqrt(0.5 * lambda / x);
    double a = b * x / mu;
    double y = std::erfc(a - b);
    y -= cdfCoef * std::erfc(a + b);
    return 0.5 * y;
  }

  RealType Variate() const override
  {
    RealType X = NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
    X *= X;
    RealType mupX = mu * X;
    RealType y = 4 * lambda + mupX;
    y = std::sqrt(y * mupX);
    y -= mupX;
    y *= -0.5 / lambda;
    ++y;
    if(U * (1 + y) > 1.0)
      y = 1.0 / y;
    return mu * y;
  }

  long double Mean() const override
  {
    return mu;
  }

  long double Variance() const override
  {
    return mu * mu * mu / lambda;
  }

  RealType Mode() const override
  {
    RealType aux = 1.5 * mu / lambda;
    RealType mode = 1 + aux * aux;
    mode = std::sqrt(mode);
    mode -= aux;
    return mu * mode;
  }

  long double Skewness() const override
  {
    return 3 * std::sqrt(mu / lambda);
  }

  long double ExcessKurtosis() const override
  {
    return 15 * mu / lambda;
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
    double im = mu * mu;
    im *= t / lambda;
    std::complex<double> y(1, -im - im);
    y = 1.0 - std::sqrt(y);
    y *= lambda / mu;
    return std::exp(y);
  }
};
} // namespace randlib
