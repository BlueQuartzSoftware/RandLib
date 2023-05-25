#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"
#include "distributions/univariate/discrete/BernoulliRand.hpp"

namespace randlib
{
/**
 * @brief The CantorRand class <BR>
 * Cantor distribution
 *
 * Notation X ~ Cantor()
 */
class RANDLIB_EXPORT CantorRand : public randlib::SingularDistribution
{
  static constexpr int n = 30;
  static double table[n]; /// all powers of 1/3 from 1 to n

  static bool SetupTable()
  {
    table[0] = 0.33333333333333333333;
    for(int i = 1; i != n; ++i)
      table[i] = table[i - 1] / 3.0;
    return true;
  }

public:
  CantorRand()
  {
    table[CantorRand::n] = {0};
    SetupTable();
  }

  String Name() const override
  {
    return "Cantor";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  double MinValue() const override
  {
    return 0;
  }

  double MaxValue() const override
  {
    return 1;
  }

  double F(const double& x) const override
  {
    if(x <= 0.0)
      return 0.0;
    if(x >= 1.0)
      return 1.0;
    double a = 0.0, b = 1.0;
    double v = 0.5, d = 0.5, delta = 1.0;
    while(delta > MIN_POSITIVE)
    {
      delta = (b - a) / 3.0;
      if(x < a + delta)
      {
        b = a + delta;
        d *= 0.5;
        v -= d;
      }
      else if(x > b - delta)
      {
        a = b - delta;
        d *= 0.5;
        v += d;
      }
      else
        return v;
    }
    return v;
  }

  double Variate() const override
  {
    long double sum = 0.0;
    for(int i = 0; i != n; ++i)
    {
      sum += table[i] * BernoulliRand::StandardVariate(this->localRandGenerator);
    }
    return sum + sum;
  }

  long double Mean() const override
  {
    return 0.5;
  }

  long double Variance() const override
  {
    return 0.125;
  }

  double Median() const override
  {
    return 1.0 / 3;
  }

  long double Skewness() const override
  {
    return 0.0l;
  }

  long double ExcessKurtosis() const override
  {
    return -1.6l;
  }

private:
  double quantileImpl(double p, double initValue) const override
  {
    if(!RandMath::findRootBrentFirstOrder<double>([this, p](double x) { return F(x) - p; }, 0.0, 1.0, initValue))
      throw std::runtime_error("Cantor distribution: failure in numerical procedure");
    return initValue;
  }

  double quantileImpl(double p) const override
  {
    return quantileImpl(p, p);
  }

  double quantileImpl1m(double p, double initValue) const override
  {
    if(!RandMath::findRootBrentFirstOrder<double>([this, p](double x) { return S(x) - p; }, 0.0, 1.0, initValue))
      throw std::runtime_error("Cantor distribution: failure in numerical procedure");
    return initValue;
  }

  double quantileImpl1m(double p) const override
  {
    return quantileImpl1m(p, 1.0 - p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    double prod = 1.0;
    for(int i = 0; i != n; ++i)
      prod *= std::cos(table[i]);
    std::complex<double> y(0.0, 0.5 * t);
    y = std::exp(y);
    return y * prod;
  }
};
} // namespace randlib
