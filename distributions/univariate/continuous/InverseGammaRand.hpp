#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/GammaRand.hpp"

namespace randlib
{
/**
 * @brief The InverseGammaRand class <BR>
 * Inverse-Gamma distribution
 *
 * X ~ Inv-Γ(α, β)
 *
 * Related distributions: <BR>
 * X = 1 / Y, where Y ~ Gamma(α, β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT InverseGammaRand : public randlib::ContinuousDistribution<RealType>
{
  double alpha = 1;   ///< shape α
  double beta = 1;    ///< rate β
  double pdfCoef = 0; ///< coefficient for faster pdf calculation

  randlib::GammaRand<RealType> X{};

public:
  InverseGammaRand(double shape = 1, double rate = 1)
  {
    SetParameters(shape, rate);
  }

  String Name() const override
  {
    return "Inverse-Gamma(" + this->toStringWithPrecision(GetShape()) + ", " + this->toStringWithPrecision(GetRate()) + ")";
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

  void SetParameters(double shape, double rate)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Inverse-Gamma distribution: shape should be positive");
    if(rate <= 0.0)
      throw std::invalid_argument("Inverse-Gamma distribution: rate should be positive");
    X.SetParameters(shape, rate);
    alpha = X.GetShape();
    beta = X.GetRate();
    pdfCoef = alpha * X.GetLogRate() - X.GetLogGammaShape();
  }

  inline double GetShape() const
  {
    return alpha;
  }

  inline double GetRate() const
  {
    return beta;
  }

  inline double GetLogShape() const
  {
    return X.GetLogShape();
  }

  inline double GetLogRate() const
  {
    return X.GetLogRate();
  }

  double logf(const RealType& x) const override
  {
    if(x <= 0.0)
      return -INFINITY;
    double logX = std::log(x);
    double y = -(alpha - 1.0) * logX;
    y -= beta / x;
    y += pdfCoef;
    return y - 2 * logX;
  }

  double F(const RealType& x) const override
  {
    return (x > 0.0) ? X.S(1.0 / x) : 0.0;
  }

  double S(const RealType& x) const override
  {
    return (x > 0.0) ? X.F(1.0 / x) : 1.0;
  }

  RealType Variate() const override
  {
    return 1.0 / X.Variate();
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    X.Sample(outputData);
    for(RealType& var : outputData)
      var = 1.0 / var;
  }

  void Reseed(unsigned long seed) const override
  {
    X.Reseed(seed);
  }

  long double Mean() const override
  {
    return (alpha > 1) ? beta / (alpha - 1) : INFINITY;
  }

  long double Variance() const override
  {
    if(alpha <= 2)
      return INFINITY;
    double var = beta / (alpha - 1);
    var *= var;
    return var / (alpha - 2);
  }

  RealType Median() const override
  {
    return 1.0 / X.Median();
  }

  RealType Mode() const override
  {
    return beta / (alpha + 1);
  }

  long double Skewness() const override
  {
    return (alpha > 3) ? 4 * std::sqrt(alpha - 2) / (alpha - 3) : INFINITY;
  }

  long double ExcessKurtosis() const override
  {
    if(alpha <= 4)
      return INFINITY;
    long double numerator = 30 * alpha - 66.0;
    long double denominator = (alpha - 3) * (alpha - 4);
    return numerator / denominator;
  }

  double GetLogGammaShape() const
  {
    return X.GetLogGammaShape();
  }

private:
  RealType quantileImpl(double p) const override
  {
    return 1.0 / X.Quantile1m(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    return 1.0 / X.Quantile(p);
  }
};
} // namespace randlib
