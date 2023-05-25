#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/discrete/ZetaRand.hpp"

namespace randlib
{
/**
 * @brief The PlanckRand class <BR>
 * Planck distribution
 *
 * f(x | a, b) = g(a + 1) * (x ^ a) / (exp(b * x) - 1)
 * where g(y) = b ^ y / (Γ(y) * ζ(y))
 *
 * Notation: X ~ Planck(a, b)
 *
 * Related distributions: <BR>
 * If G ~ Gamma(a + 1, b) and Z ~ Zeta(a + 1), then G / Z ~ Planck(a, b)
 */
template <typename RealType = double>
class RANDLIB_EXPORT PlanckRand : public randlib::ContinuousDistribution<RealType>
{
  double a = 1;                                ///< shape
  double b = 1;                                ///< scale
  double pdfCoef = M_LN2 + M_LN3 - 2 * M_LNPI; ///< coefficient for faster pdf calculations

  ZetaRand<long long int> Z{};
  GammaRand<RealType> G{2};

public:
  PlanckRand(double shape = 1, double scale = 1)
  {
    SetParameters(shape, scale);
  }

  String Name() const override
  {
    return "Planck(" + this->toStringWithPrecision(GetShape()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
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

  void SetParameters(double shape, double scale)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Planck distribution: shape should be positive, but it's equal to " + std::to_string(shape));
    if(scale <= 0.0)
      throw std::invalid_argument("Planck distribution: scale should be positive, but it's equal to " + std::to_string(scale));
    a = shape;
    b = scale;

    double ap1 = a + 1;
    Z.SetExponent(ap1);
    G.SetParameters(ap1, b);

    pdfCoef = -Z.GetLogZetaFunction();
    pdfCoef += ap1 * std::log(b);
    pdfCoef -= G.GetLogGammaShape();
  }

  inline double GetShape() const
  {
    return a;
  }

  inline double GetScale() const
  {
    return b;
  }

  double f(const RealType& x) const override
  {
    if(x < 0)
      return 0;
    if(x == 0)
    {
      if(a > 1)
        return 0.0;
      if(a == 1)
        return std::exp(pdfCoef) / b;
      return INFINITY;
    }
    double y = pdfCoef + a * std::log(x);
    return std::exp(y) / std::expm1l(b * x);
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0)
    {
      if(a > 1)
        return -INFINITY;
      if(a == 1)
        return pdfCoef - G.GetLogRate();
      return INFINITY;
    }
    double y = pdfCoef + a * std::log(x);
    return y - RandMath::logexpm1l(b * x);
  }

  double F(const RealType& x) const override
  {
    if(x <= 0)
      return 0.0;

    if(a >= 1)
    {
      return RandMath::integral([this](double t) { return f(t); }, 0, x);
    }

    /// split F(x) by two integrals
    double aux = pdfCoef + a * std::log(x);
    double integral1 = std::exp(aux) / (b * a);
    double integral2 = RandMath::integral([this](double t) { return h(t); }, 0, x);
    return integral1 + integral2;
  }

  RealType Variate() const override
  {
    return G.Variate() / Z.Variate();
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    G.Sample(outputData);
    for(RealType& var : outputData)
      var /= Z.Variate();
  }

  long double Mean() const override
  {
    double y = (a + 1) / b;
    y *= std::riemann_zetal(a + 2);
    return y / Z.GetZetaFunction();
  }

  long double SecondMoment() const override
  {
    long double secondMoment = (a + 1) * (a + 2);
    secondMoment /= (b * b);
    secondMoment *= std::riemann_zetal(a + 3);
    secondMoment /= Z.GetZetaFunction();
    return secondMoment;
  }

  long double Variance() const override
  {
    long double mean = Mean();
    return SecondMoment() - mean * mean;
  }

  RealType Mode() const override
  {
    if(a <= 1)
      return 0.0;
    double y = -a * std::exp(-a);
    y = RandMath::W0Lambert(y);
    return (y + a) / b;
  }

  long double ThirdMoment() const override
  {
    long double thirdMoment = (a + 3) * (a + 2) * (a + 1);
    thirdMoment /= (b * b * b);
    thirdMoment *= std::riemann_zetal(a + 4);
    thirdMoment /= Z.GetZetaFunction();
    return thirdMoment;
  }

  long double Skewness() const override
  {
    long double mean = Mean();
    long double secondMoment = SecondMoment();
    long double thirdMoment = ThirdMoment();
    long double meanSq = mean * mean;
    long double variance = secondMoment - meanSq;
    long double numerator = thirdMoment - 3 * mean * variance - mean * meanSq;
    long double denominator = std::pow(variance, 1.5);
    return numerator / denominator;
  }

  long double FourthMoment() const override
  {
    long double fourthMoment = (a + 4) * (a + 3) * (a + 2) * (a + 1);
    long double bSq = b * b;
    fourthMoment /= (bSq * bSq);
    fourthMoment *= std::riemann_zetal(a + 5);
    fourthMoment /= Z.GetZetaFunction();
    return fourthMoment;
  }

  long double ExcessKurtosis() const override
  {
    long double mean = Mean();
    long double secondMoment = SecondMoment();
    long double thirdMoment = ThirdMoment();
    long double fourthMoment = FourthMoment();
    long double meanSq = mean * mean;
    long double variance = secondMoment - meanSq;
    long double numerator = fourthMoment - 4 * thirdMoment * mean + 6 * secondMoment * meanSq - 3 * meanSq * meanSq;
    long double denominator = variance * variance;
    return numerator / denominator - 3.0;
  }

private:
  /**
   * @fn h
   * @param t
   * @return leveled pdf for F and CF calculations
   */
  double h(double t) const
  {
    if(t <= 0)
      return 0.0;
    double y = pdfCoef + a * std::log(t);
    double expY = std::exp(y);
    double bt = b * t;
    double z = 1.0 / std::expm1l(bt) - 1.0 / bt;
    return expY * z;
  }

  std::complex<double> CFImpl(double t) const override
  {
    if(a >= 1)
      return randlib::ContinuousDistribution<RealType>::CFImpl(t);

    /// We have singularity point at 0 for real part,
    /// so we split the integral in two intervals:
    /// First one from 0 to 1, for which we integrate
    /// numerically leveled pdf and add known solution for level.
    /// Second one from 1 to infinity, for which we use
    /// simple expected value for the rest of the function
    double re1 = RandMath::integral([this, t](double x) { return std::cos(t * x) * h(x); }, 0.0, 1.0);

    double re2 = this->ExpectedValue([this, t](double x) { return std::cos(t * x); }, 1.0, INFINITY);

    double re3 = t * RandMath::integral(
                         [this, t](double x) {
                           if(x <= 0.0)
                             return 0.0;
                           return std::sin(t * x) * std::pow(x, a);
                         },
                         0.0, 1.0);

    re3 += std::cos(t);
    re3 *= std::exp(pdfCoef) / (b * a);

    double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, 0.0, INFINITY);

    double re = re1 + re2 + re3;
    return std::complex<double>(re, im);
  }
};
} // namespace randlib
