#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

namespace randlib
{
/**
 * @brief The DegenerateRand class <BR>
 * Degenerate distribution
 *
 * f(x|a) = δ(a)
 *
 * Notation: X ~ δ(a)
 */
class RANDLIB_EXPORT DegenerateRand : public randlib::ContinuousDistribution<double>
{
  double a; ///< value

public:
  explicit DegenerateRand(double value = 0)
  : a(value)
  {
  }

  String Name() const override
  {
    return "Degenerate";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  double MinValue() const override
  {
    return a;
  }

  double MaxValue() const override
  {
    return a;
  }

  void SetValue(double value)
  {
    a = value;
  }

  inline double GetValue() const
  {
    return a;
  }

  double f(const double& x) const override
  {
    return (x == a) ? INFINITY : 0.0;
  }

  double logf(const double& x) const override
  {
    return (x == a) ? INFINITY : -INFINITY;
  }

  double F(const double& x) const override
  {
    return (x < a) ? 0.0 : 1.0;
  }

  double Variate() const override
  {
    return a;
  }

  long double Mean() const override
  {
    return a;
  }

  long double Variance() const override
  {
    return 0.0;
  }

  double Median() const override
  {
    return a;
  }

  double Mode() const override
  {
    return a;
  }

  long double Skewness() const override
  {
    return NAN;
  }

  long double ExcessKurtosis() const override
  {
    return NAN;
  }

  long double Entropy() const
  {
    return 0.0;
  }

  /**
   * @fn Fit
   * @param sample
   */
  void Fit(const std::vector<double>& sample)
  {
    auto sampleBegin = sample.begin();
    if(!std::equal(sampleBegin, sample.end(), sampleBegin))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, "All elements should be equal to each other"));
    SetValue(*sampleBegin);
  }

private:
  double quantileImpl(double p) const override
  {
    return a;
  }

  double quantileImpl1m(double p) const override
  {
    return a;
  }

  std::complex<double> CFImpl(double t) const override
  {
    double re = std::cos(a * t);
    double im = std::sin(a * t);
    return std::complex<double>(re, im);
  }
};
} // namespace randlib
