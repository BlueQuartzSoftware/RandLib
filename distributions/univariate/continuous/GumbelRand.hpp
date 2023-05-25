#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"

#include "external/log.hpp"

namespace randlib
{
/**
 * @brief The GumbelRand class <BR>
 * Gumbel distribution
 *
 * Notation: X ~ Gumbel(μ, β)
 *
 * Related distributions: <BR>
 * exp(-(X - μ) / β) ~ Exp(1)
 */
template <typename RealType = double>
class RANDLIB_EXPORT GumbelRand : public randlib::ContinuousDistribution<RealType>
{
  double mu = 0;      ///< location μ
  double beta = 1;    ///< scale β
  double logBeta = 0; ///< log(β)
public:
  GumbelRand(double location = 0, double scale = 1)
  {
    SetLocation(location);
    SetScale(scale);
  }

  String Name() const override
  {
    return "Gumbel(" + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
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

  void SetLocation(double location)
  {
    mu = location;
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Gumbel distribution: scale should be positive");
    beta = scale;
    logBeta = std::log(beta);
  }

  inline double GetLocation() const
  {
    return mu;
  }

  inline double GetScale() const
  {
    return beta;
  }

  double f(const RealType& x) const override
  {
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    double z = (mu - x) / beta;
    double y = std::exp(z);
    return z - y - logBeta;
  }

  double F(const RealType& x) const override
  {
    double y = (mu - x) / beta;
    y = std::exp(y);
    return std::exp(-y);
  }

  double S(const RealType& x) const override
  {
    double y = (mu - x) / beta;
    y = std::exp(y);
    return -std::expm1l(-y);
  }

  RealType Variate() const override
  {
    return mu + beta * GumbelRand<RealType>::StandardVariate(this->localRandGenerator);
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    RealType w = ExponentialRand<RealType>::StandardVariate(randGenerator);
    return -std::log(w);
  }

  long double Mean() const override
  {
    return mu + beta * M_EULER;
  }

  long double Variance() const override
  {
    double v = M_PI * beta;
    return v * v / 6;
  }

  RealType Median() const override
  {
    static constexpr double M_LN_LN2 = nonstd::log(M_LN2);
    return mu - beta * M_LN_LN2;
  }

  RealType Mode() const override
  {
    return mu;
  }

  long double Skewness() const override
  {
    static constexpr long double skew = 12 * M_SQRT2 * M_SQRT3 * M_APERY / (M_PI_SQ * M_PI);
    return skew;
  }

  long double ExcessKurtosis() const override
  {
    return 2.4l;
  }

  long double Entropy() const
  {
    return logBeta + M_EULER + 1.0;
  }

private:
  RealType quantileImpl(double p) const override
  {
    return mu - beta * std::log(-std::log(p));
  }

  RealType quantileImpl1m(double p) const override
  {
    return mu - beta * std::log(-std::log1pl(-p));
  }
};
} // namespace randlib
