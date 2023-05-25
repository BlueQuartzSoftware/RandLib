#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The VonMisesRand class <BR>
 * Von-Mises distribution
 *
 * Notation: X ~ Von-Mises(μ, k)
 */
template <typename RealType = double>
class RANDLIB_EXPORT VonMisesRand : public randlib::CircularDistribution<RealType>
{
  double k = 1;      ///< concentration
  double logI0k = 1; ///< log(I_0(k)), where I_0 stands for modified Bessel
  ///< function of the first kind of order 0
  double s = M_PI / M_E; ///< generator coefficient
  int p = 12;            /// coefficient for faster cdf calculation
  static constexpr double CK = 10.5;

public:
  VonMisesRand(double location = 0, double concentration = 1)
  {
    SetConcentration(concentration);
  }

  String Name() const override
  {
    return "von Mises(" + this->toStringWithPrecision(this->GetLocation()) + ", " + this->toStringWithPrecision(this->GetConcentration()) + ")";
  }

  void SetConcentration(double concentration)
  {
    if(concentration <= 0.0)
      throw std::invalid_argument("von Mises distribution: concentration parameter should be positive");
    k = concentration;
    logI0k = RandMath::logBesselI(0, k);
    s = (k > 1.3) ? 1.0 / std::sqrt(k) : M_PI * std::exp(-k);
    if(k < CK)
    {
      /// set coefficients for cdf
      static constexpr double a[] = {28.0, 0.5, 100.0, 5.0};
      p = std::ceil(a[0] + a[1] * k - a[2] / (k + a[3]));
    }
  }

  inline double GetConcentration() const
  {
    return k;
  }

  double f(const RealType& x) const override
  {
    return (x < this->loc - M_PI || x > this->loc + M_PI) ? 0.0 : std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < this->loc - M_PI || x > this->loc + M_PI)
      return -INFINITY;
    double y = k * std::cos(x - this->loc) - logI0k;
    y -= M_LNPI + M_LN2;
    return y;
  }

  double F(const RealType& x) const override
  {
    if(x <= this->loc - M_PI)
      return 0.0;
    if(x >= this->loc + M_PI)
      return 1.0;
    double xAdj = x - this->loc;
    xAdj -= M_2_PI * std::round(0.5 * xAdj / M_PI);
    return (k < CK) ? cdfSeries(xAdj) : cdfErfc(xAdj);
  }

  double S(const RealType& x) const override
  {
    if(x <= this->loc - M_PI)
      return 1.0;
    if(x >= this->loc + M_PI)
      return 0.0;
    double xAdj = x - this->loc;
    xAdj -= M_2_PI * std::round(0.5 * xAdj / M_PI);
    return (k < CK) ? 1.0 - cdfSeries(xAdj) : ccdfErfc(xAdj);
  }

  RealType Variate() const override
  {
    /// Generating von Mises variates by the ratio-of-uniforms method
    /// Lucio Barabesi. Dipartimento di Metodi Quantitativi, Universiteta di Siena
    size_t iter = 0;
    do
    {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType V = 2 * UniformRand<RealType>::StandardVariate(this->localRandGenerator) - 1;
      RealType theta = s * V / U;
      if((std::fabs(theta) <= M_PI) && ((k * theta * theta < 4.0 * (1.0 - U)) || (k * std::cos(theta) >= 2 * std::log(U) + k)))
        return this->loc + theta;
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("von Mises distribution: sampling failed");
  }

  long double CircularMean() const override
  {
    return this->loc;
  }

  long double CircularVariance() const override
  {
    double var = RandMath::logBesselI(1, k);
    var -= logI0k;
    return -std::expm1l(var);
  }

  RealType Median() const override
  {
    return this->loc;
  }

  RealType Mode() const override
  {
    return this->loc;
  }

private:
  double cdfSeries(double x) const
  {
    /// backwards recursion
    double sinX = std::sin(x), cosX = std::cos(x);
    double px = p * x;
    double sn = std::sin(px), cn = std::cos(px);
    double R = 0, V = 0;
    for(int n = p - 1; n > 0; --n)
    {
      double temp = sn;
      sn *= cosX;
      sn -= cn * sinX;
      cn *= cosX;
      cn += temp * sinX;
      R += 2 * n / k;
      R = 1.0 / R;
      V += sn / n;
      V *= R;
    }
    V /= M_PI;
    V += 0.5 * x / M_PI;
    V += 0.5;
    return V;
  }

  double cdfErfcAux(RealType x) const
  {
    /// Normal cdf approximation
    double c = 24.0 * k;
    double v = c - 56.0;
    double r = std::sqrt((54.0 / (347.0 / v + 26.0 - c) - 6.0 + c) / 12.0);
    double z = -std::sin(0.5 * x) * r;
    double twoZSq = 2.0 * z * z;
    v -= twoZSq - 3.0;
    double y = (c - 2 * twoZSq - 16.0) / 3.0;
    y = ((twoZSq + 1.75) * twoZSq + 83.5) / v - y;
    y *= y;
    return z * (1.0 - twoZSq / y);
  }

  double cdfErfc(RealType x) const
  {
    double arg = cdfErfcAux(x);
    return 0.5 * std::erfc(arg);
  }

  double ccdfErfc(RealType x) const
  {
    double arg = cdfErfcAux(x);
    return 0.5 * std::erfc(-arg);
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
    double tloc = t * this->loc;
    double cosTloc = std::cos(tloc), sinTloc = std::sin(tloc);
    std::complex<double> y(cosTloc, sinTloc);
    double z = RandMath::logBesselI(t, k) - logI0k;
    return y * std::exp(z);
  }
};
} // namespace randlib
