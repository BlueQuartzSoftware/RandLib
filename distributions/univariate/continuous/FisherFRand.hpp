#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/BetaPrimeRand.hpp"

namespace randlib
{
/**
 * @brief The FisherFRand class <BR>
 * F-distribution
 *
 * Notation: X ~ F(d1, d2)
 *
 * Related distributions: <BR>
 * d1/d2 * X ~ B'(d1/2, d2/2)
 */
template <typename RealType = double>
class RANDLIB_EXPORT FisherFRand : public randlib::ContinuousDistribution<RealType>
{
  int d1 = 2;         ///< first degree
  int d2 = 2;         ///< second degree
  double a = 0;       ///< d1 / 2 - 1;
  double d1_d2 = 1;   ///< d1 / d2
  double c = -2;      ///< -(d1 + d2) / 2;
  double d2_d1 = 1;   ///< d2 / d1
  double pdfCoef = 0; /// < (a + 1) * log(d1/d2) - log(B(d1/2, d2/2))

  BetaPrimeRand<RealType> B{};

public:
  FisherFRand(int degree1 = 2, int degree2 = 2)
  {
    SetDegrees(degree1, degree2);
  }

  String Name() const override
  {
    return "Fisher-F(" + this->toStringWithPrecision(GetFirstDegree()) + ", " + this->toStringWithPrecision(GetSecondDegree()) + ")";
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

  void SetDegrees(int degree1, int degree2)
  {
    if(degree1 <= 0 || degree2 <= 0)
      throw std::invalid_argument("F-distribution: degrees of should be positive");

    d1 = degree1;
    d2 = degree2;

    B.SetShapes(0.5 * d1, 0.5 * d2);

    a = 0.5 * d1 - 1;
    d1_d2 = static_cast<double>(d1) / d2;
    c = -0.5 * (d1 + d2);
    d2_d1 = 1.0 / d1_d2;

    pdfCoef = (a + 1) * std::log(d1_d2);
    pdfCoef -= B.GetLogBetaFunction();
  }

  inline int GetFirstDegree() const
  {
    return d1;
  }

  inline int GetSecondDegree() const
  {
    return d2;
  }

  double f(const RealType& x) const override
  {
    if(x < 0.0)
      return 0.0;
    if(x == 0.0)
    {
      if(a == 0.0)
        return std::exp(pdfCoef);
      return (a > 0) ? 0.0 : INFINITY;
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0.0)
    {
      if(a == 0.0)
        return pdfCoef;
      return (a > 0) ? -INFINITY : INFINITY;
    }
    double y = a * std::log(x);
    y += c * std::log1pl(d1_d2 * x);
    return pdfCoef + y;
  }

  double F(const RealType& x) const override
  {
    return B.F(d1_d2 * x);
  }

  double S(const RealType& x) const override
  {
    return B.S(d1_d2 * x);
  }

  RealType Variate() const override
  {
    return d2_d1 * B.Variate();
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    B.Sample(outputData);
    for(RealType& var : outputData)
      var = d2_d1 * var;
  }

  void Reseed(unsigned long seed) const override
  {
    B.Reseed(seed);
  }

  long double Mean() const override
  {
    return (d2 > 2) ? 1 + 2.0 / (d2 - 2) : INFINITY;
  }

  long double Variance() const override
  {
    if(d2 <= 4)
      return INFINITY;
    double variance = d2;
    variance /= d2 - 2;
    variance *= variance;
    variance *= 2 * (d1 + d2 - 2);
    variance /= d1;
    variance /= d2 - 4;
    return variance;
  }

  RealType Median() const override
  {
    return d2_d1 * B.Median();
  }

  RealType Mode() const override
  {
    if(d1 <= 2)
      return 0.0;
    return d2_d1 * (d1 - 2) / (d2 + 2);
  }

  long double Skewness() const override
  {
    if(d2 <= 6)
      return INFINITY;
    long double skewness = 8.0 * (d2 - 4.0);
    long double aux = d1 + d2 - 2;
    skewness /= d1 * aux;
    skewness = std::sqrt(skewness);
    skewness *= d1 + aux;
    skewness /= d2 - 6.0;
    return skewness;
  }

  long double ExcessKurtosis() const override
  {
    if(d2 <= 8)
      return INFINITY;
    long double kurtosis = d2 - 2;
    kurtosis *= kurtosis;
    kurtosis *= d2 - 4;
    kurtosis /= d1;
    kurtosis /= d1 + d2 - 2;
    kurtosis += 5 * d2 - 22;
    kurtosis /= d2 - 6;
    kurtosis /= d2 - 8;
    return 12.0 * kurtosis;
  }

private:
  RealType quantileImpl(double p) const override
  {
    return d2_d1 * B.Quantile(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    return d2_d1 * B.Quantile1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    return B.CF(d2_d1 * t);
  }
};
} // namespace randlib
