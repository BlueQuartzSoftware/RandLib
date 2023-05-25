#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"

namespace randlib
{
/**
 * @brief The FrechetRand class <BR>
 * Frechet distribution
 *
 * Notation: X ~ Frechet(α, s, m)
 */
template <typename RealType = double>
class RANDLIB_EXPORT FrechetRand : public randlib::ContinuousDistribution<RealType>
{
  double alpha = 1;    ///< shape α
  double s = 1;        ///< scale
  double m = 0;        ///< location
  double alphaInv = 1; ///< 1/α
  double pdfCoef = 0;  ///< log(α/s)

public:
  FrechetRand(double shape = 1, double scale = 1, double location = 0)
  {
    SetParameters(shape, scale, location);
  }

  String Name() const override
  {
    return "Frechet(" + this->toStringWithPrecision(GetShape()) + ", " + this->toStringWithPrecision(GetScale()) + ", " + this->toStringWithPrecision(GetLocation()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  RealType MinValue() const override
  {
    return m;
  }

  RealType MaxValue() const override
  {
    return INFINITY;
  }

  void SetParameters(double shape, double scale, double location)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Frechet distribution: shape should be positive");
    if(scale <= 0.0)
      throw std::invalid_argument("Frechet distribution: scale should be positive");
    alpha = shape;
    alphaInv = 1.0 / alpha;
    s = scale;
    pdfCoef = std::log(alpha / s);
    m = location;
  }

  inline double GetShape() const
  {
    return alpha;
  }

  inline double GetScale() const
  {
    return s;
  }

  inline double GetLocation() const
  {
    return m;
  }

  double f(const RealType& x) const override
  {
    return (x <= m) ? 0.0 : std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x <= m)
      return -INFINITY;
    double xAdj = (x - m) / s;
    double logxAdj = std::log(xAdj);
    double a = alpha * logxAdj;
    double expA = std::exp(-a);
    return pdfCoef - a - expA - logxAdj;
  }

  double F(const RealType& x) const override
  {
    if(x <= m)
      return 0.0;
    double xAdj = (x - m) / s;
    double xPow = std::pow(xAdj, -alpha);
    return std::exp(-xPow);
  }

  double S(const RealType& x) const override
  {
    if(x <= m)
      return 1.0;
    double xAdj = (x - m) / s;
    double xPow = std::pow(xAdj, -alpha);
    return -std::expm1l(-xPow);
  }

  RealType Variate() const override
  {
    return m + s / std::pow(ExponentialRand<RealType>::StandardVariate(this->localRandGenerator), alphaInv);
  }

  long double Mean() const override
  {
    if(alpha <= 1.0)
      return INFINITY;
    return m + s * std::tgammal(1.0 - alphaInv);
  }

  long double Variance() const override
  {
    if(alpha <= 2.0)
      return INFINITY;
    long double var = std::tgammal(1.0 - alphaInv);
    var *= var;
    var = std::tgammal(1.0 - 2 * alphaInv) - var;
    return s * s * var;
  }

  RealType Median() const override
  {
    return m + s / std::pow(M_LN2, alphaInv);
  }

  RealType Mode() const override
  {
    RealType y = alpha / (1.0 + alpha);
    y = std::pow(y, alphaInv);
    return m + s * y;
  }

  long double Skewness() const override
  {
    if(alpha <= 3.0)
      return INFINITY;
    long double x = std::tgammal(1.0 - alphaInv);
    long double y = std::tgammal(1.0 - 2.0 * alphaInv);
    long double z = std::tgammal(1.0 - 3.0 * alphaInv);
    long double numerator = 2 * x * x - 3 * y;
    numerator *= x;
    numerator += z;
    long double denominator = y - x * x;
    denominator = std::pow(denominator, 1.5);
    return numerator / denominator;
  }

  long double ExcessKurtosis() const override
  {
    if(alpha <= 4.0)
      return INFINITY;
    long double x = std::tgammal(1.0 - alphaInv);
    long double y = std::tgammal(1.0 - 2.0 * alphaInv);
    long double z = std::tgammal(1.0 - 3.0 * alphaInv);
    long double w = std::tgammal(1.0 - 4.0 * alphaInv);
    long double numerator = w - 4 * z * x + 3 * y * y;
    long double denominator = y - x * x;
    denominator *= denominator;
    return numerator / denominator - 6.0;
  }

  long double Entropy() const
  {
    return 1.0 + M_EULER * (1.0 + alphaInv) + std::log(s / alpha);
  }

private:
  RealType quantileImpl(double p) const override
  {
    RealType y = -std::log(p);
    y = s / std::pow(y, alphaInv);
    return y + m;
  }

  RealType quantileImpl1m(double p) const override
  {
    RealType y = -std::log1pl(-p);
    y = s / std::pow(y, alphaInv);
    return y + m;
  }
};
} // namespace randlib
