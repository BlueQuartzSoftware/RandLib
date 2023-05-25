#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/GeometricRand.hpp"

namespace randlib
{
/**
 * @brief The YuleRand class <BR>
 * Yule distribution
 *
 * Notation: X ~ Yule(ρ)
 *
 * Related distributions: <BR>
 * If Y ~ Pareto(ρ, 1) and Z ~ Geometric(1 / Y), then Z + 1 ~ Yule(ρ)
 */
template <typename IntType = int>
class RANDLIB_EXPORT YuleRand : public randlib::DiscreteDistribution<IntType>
{
  double rho = 0;        ///< shape ρ
  double lgamma1pRo = 0; /// log(Γ(1 + ρ))

  ParetoRand<double> X;

public:
  explicit YuleRand(double shape)
  : X(shape, 1.0)
  {
    SetShape(shape);
  }

  String Name() const override
  {
    return "Yule(" + this->toStringWithPrecision(GetShape()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  IntType MinValue() const override
  {
    return 1;
  }

  IntType MaxValue() const override
  {
    return std::numeric_limits<IntType>::max();
  }

  void SetShape(double shape)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Yule distribution: shape should be positive");
    rho = shape;
    lgamma1pRo = std::lgammal(rho + 1);
    X.SetShape(rho);
  }

  inline double GetShape() const
  {
    return rho;
  }

  double logP(const IntType& k) const override
  {
    if(k < 1)
      return -INFINITY;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + rho + 1);
    y += X.GetLogShape();
    return y;
  }

  double F(const IntType& k) const override
  {
    if(k < 1)
      return 0.0;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + rho + 1);
    double logk = std::log(k);
    return -std::expm1(y + logk);
  }

  double S(const IntType& k) const override
  {
    if(k < 1)
      return 1.0;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + rho + 1);
    y = std::exp(y);
    return k * y;
  }

  IntType Variate() const override
  {
    double prob = 1.0 / X.Variate();
    return GeometricRand<IntType>::Variate(prob, this->localRandGenerator) + 1;
  }

  static IntType Variate(double shape, RandGenerator& randGenerator = ProbabilityDistribution<IntType>::staticRandGenerator)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Yule distribution: shape should be positive");
    double prob = 1.0 / ParetoRand<double>::StandardVariate(shape, randGenerator);
    return GeometricRand<IntType>::Variate(prob, randGenerator) + 1;
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    X.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return (rho <= 1) ? INFINITY : rho / (rho - 1);
  }

  long double Variance() const override
  {
    if(rho <= 2)
      return INFINITY;
    double aux = rho / (rho - 1);
    return aux * aux / (rho - 2);
  }

  IntType Mode() const override
  {
    return 1;
  }

  long double Skewness() const override
  {
    if(rho <= 3)
      return INFINITY;
    long double skewness = rho + 1;
    skewness *= skewness;
    skewness *= std::sqrt(rho - 2);
    return skewness / (rho * (rho - 3));
  }

  long double ExcessKurtosis() const override
  {
    if(rho <= 4)
      return INFINITY;
    long double numerator = 11 * rho * rho - 49;
    numerator *= rho;
    numerator -= 22;
    long double denominator = rho * (rho - 4) * (rho - 3);
    return rho + 3 + numerator / denominator;
  }
};
} // namespace randlib
