#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The WrappedExponentialRand class <BR>
 * Wrapped Exponential distribution
 *
 * Notation: X ~ WE(λ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT WrappedExponentialRand : public randlib::CircularDistribution<RealType>
{
  double lambda = 1;              ///< rate λ
  double logLambda = 0;           ///< ln(λ)
  double scaledLambda = 2 * M_PI; ///< 2πλ
  double pdfCoef = 0, logpdfCoef = 0, expmScaledLambda = 0;

public:
  WrappedExponentialRand(double rate)
  {
    SetRate(rate);
  }

  String Name() const override
  {
    return "Wrapped Exponential(" + this->toStringWithPrecision(GetRate()) + ")";
  }

  void SetRate(double rate)
  {
    if(lambda <= 0.0)
      throw std::invalid_argument("Wrapped Exponential distribution: rate parameter should be positive");
    lambda = rate;
    logLambda = std::log(lambda);
    scaledLambda = 2 * M_PI * lambda;
    expmScaledLambda = std::exp(-scaledLambda);
    pdfCoef = -std::expm1l(-scaledLambda);
    logpdfCoef = RandMath::log1mexp(-scaledLambda);
  }

  inline double GetRate() const
  {
    return lambda;
  }

  double f(const RealType& x) const override
  {
    return (x < 0 || x > 2 * M_PI) ? 0.0 : std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    return (x < 0 || x > 2 * M_PI) ? -INFINITY : logLambda - lambda * x - logpdfCoef;
  }

  double F(const RealType& x) const override
  {
    if(x <= 0.0)
      return 0.0;
    return (x < 2 * M_PI) ? std::exp(RandMath::log1mexp(-lambda * x) - logpdfCoef) : 1.0;
  }

  double S(const RealType& x) const override
  {
    if(x <= 0.0)
      return 1.0;
    if(x >= 2 * M_PI)
      return 0.0;
    double y = std::expm1l(scaledLambda - lambda * x);
    y /= pdfCoef;
    return expmScaledLambda * y;
  }

  RealType Variate() const override
  {
    return quantileImpl(UniformRand<RealType>::StandardVariate(this->localRandGenerator));
  }

  long double CircularMean() const override
  {
    return M_PI_2 - RandMath::atan(lambda);
  }

  long double CircularVariance() const override
  {
    return 1.0 - 1.0 / std::sqrt(1.0 + lambda * lambda);
  }

  RealType Median() const override
  {
    return (M_LN2 - RandMath::softplus(-scaledLambda)) / lambda;
  }

  RealType Mode() const override
  {
    return 0.0;
  }

protected:
  RealType quantileImpl(double p) const override
  {
    return -std::log1pl(-p * pdfCoef) / lambda;
  }

  RealType quantileImpl1m(double p) const override
  {
    return -std::log(expmScaledLambda + p * pdfCoef) / lambda;
  }

  std::complex<double> CFImpl(double t) const override
  {
    double temp = t / lambda;
    double coef = 1.0 / (1.0 + temp * temp);
    return std::complex<double>(coef, temp * coef);
  }
};
} // namespace randlib
