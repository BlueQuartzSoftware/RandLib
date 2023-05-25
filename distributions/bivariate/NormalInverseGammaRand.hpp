#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/InverseGammaRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/StudentTRand.hpp"

namespace randlib
{
/**
 * @brief The NormalInverseGammaRand class <BR>
 * Normal-inverse-gamma distribution
 *
 * Notation: X ~ NIG(μ, λ, α, β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT NormalInverseGammaRand : public randlib::ContinuousBivariateDistribution<StudentTRand<RealType>, InverseGammaRand<RealType>, RealType>
{
  double mu = 0;                           ///< location μ
  double lambda = 1;                       ///< precision λ
  double alpha = 1;                        ///< shape α
  double beta = 1;                         ///< rate β
  double pdfCoef = 0.5 * (M_LNPI - M_LN2); ///< coefficient for faster pdf calculation

public:
  NormalInverseGammaRand(double location = 0, double precision = 1, double shape = 1, double rate = 1)
  {
    SetParameters(location, precision, shape, rate);
  }

  String Name() const override
  {
    return "Normal-Inverse-Gamma(" + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(GetPrecision()) + ", " + this->toStringWithPrecision(GetShape()) + ", " +
           this->toStringWithPrecision(GetRate()) + ")";
  }

  void SetParameters(double location, double precision, double shape, double rate)
  {
    if(precision <= 0.0)
      throw std::invalid_argument("Precision of Normal-Inverse-Gamma distribution should be positive, "
                                  "but it's equal to " +
                                  std::to_string(precision));
    if(shape <= 0.0)
      throw std::invalid_argument("Shape of Normal-Inverse-Gamma distribution "
                                  "should be positive, but it's equal to " +
                                  std::to_string(shape));
    if(rate <= 0.0)
      throw std::invalid_argument("Rate of Normal-Inverse-Gamma distribution "
                                  "should be positive, but it's equal to " +
                                  std::to_string(rate));

    mu = location;
    lambda = precision;

    this->Y.SetParameters(shape, rate);
    alpha = this->Y.GetShape();
    beta = this->Y.GetRate();
    this->X.SetDegree(2 * alpha);
    this->X.SetLocation(mu);
    this->X.SetScale(std::sqrt(alpha * lambda / beta));

    pdfCoef = 0.5 * std::log(0.5 * lambda / M_PI);
    pdfCoef += alpha * this->Y.GetLogRate() - this->Y.GetLogGammaShape();
  }

  inline double GetLocation() const
  {
    return mu;
  }

  inline double GetPrecision() const
  {
    return lambda;
  }

  inline double GetShape() const
  {
    return alpha;
  }

  inline double GetRate() const
  {
    return beta;
  }

  double f(const Pair<RealType>& point) const override
  {
    return (point.second > 0.0) ? std::exp(logf(point)) : 0.0;
  }

  double logf(const Pair<RealType>& point) const override
  {
    RealType sigmaSq = point.second;
    if(sigmaSq <= 0)
      return -INFINITY;
    RealType x = point.first;
    RealType y = (alpha + 1.5) * std::log(sigmaSq);
    RealType degree = x - mu;
    degree *= degree;
    degree *= lambda;
    degree += 2 * beta;
    degree *= 0.5 / sigmaSq;
    y += degree;
    return pdfCoef - y;
  }

  double F(const Pair<RealType>& point) const override
  {
    RealType sigmaSq = point.second;
    if(sigmaSq <= 0)
      return 0.0;
    RealType x = point.first;
    RealType y = 0.5 * lambda;
    RealType xmmu = x - mu;
    y *= xmmu * xmmu / sigmaSq;
    y = std::erfc(-std::sqrt(y));
    RealType z = beta / sigmaSq;
    RealType temp = alpha * std::log(z) - z;
    y *= std::exp(temp - this->Y.GetLogGammaShape());
    y *= 0.5 / sigmaSq;
    return y;
  }

  Pair<RealType> Variate() const override
  {
    Pair<RealType> var;
    var.second = this->Y.Variate();
    double coef = std::sqrt(var.second) / lambda;
    var.first = mu + coef * randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    return var;
  }

  long double Correlation() const override
  {
    return 0.0;
  }

  Pair<RealType> Mode() const override
  {
    return std::make_pair(mu, 2 * beta / (2 * alpha + 3));
  }
};
} // namespace randlib