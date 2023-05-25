#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/StableRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The CauchyRand class <BR>
 * Cauchy distribution
 *
 * f(x | μ, σ) = σ / [π (σ^2 + (x - μ)^2)]
 *
 * Notation: X ~ Cauchy(μ, σ)
 *
 * Related distributions: <BR>
 * If X ~ Cauchy(0, 1), then μ + σ * X ~ Cauchy(μ, σ) <BR>
 * X ~ S(1, 0, σ, μ) <BR>
 * If X, Y ~ Normal(0, 1), then X / Y ~ Cauchy(0, 1)
 */
template <typename RealType = double>
class RANDLIB_EXPORT CauchyRand : public StableDistribution<RealType>
{
public:
  CauchyRand(double location = 0, double scale = 1)
  : StableDistribution<RealType>(1, 0, scale, location)
  {
  }

  String Name() const override
  {
    return "Cauchy(" + this->toStringWithPrecision(this->GetLocation()) + ", " + this->toStringWithPrecision(this->GetScale()) + ")";
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

  double f(const RealType& x) const override
  {
    return this->pdfCauchy(x);
  }

  double F(const RealType& x) const override
  {
    return this->cdfCauchy(x);
  }

  double S(const RealType& x) const override
  {
    return this->cdfCauchyCompl(x);
  }

  RealType Variate() const override
  {
    return this->mu + this->gamma * StandardVariate(this->localRandGenerator);
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    double x, y;
    do
    {
      x = 2 * UniformRand<RealType>::StandardVariate(randGenerator) - 1;
      y = 2 * UniformRand<RealType>::StandardVariate(randGenerator) - 1;
    } while(y == 0.0 || x * x + y * y > 1.0);
    return x / y;
  }

  long double Entropy() const
  {
    return 2 * M_LN2 + this->logGamma + M_LNPI;
  }

private:
  RealType quantileImpl(double p) const override
  {
    return this->quantileCauchy(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    return this->quantileCauchy1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    return this->cfCauchy(t);
  }
};
} // namespace randlib
