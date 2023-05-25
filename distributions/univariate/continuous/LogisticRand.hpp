#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The LogisticRand class <BR>
 * Logistic distribution
 *
 * Notation: X ~ Logistic(μ, s)
 *
 * Related distributions: <BR>
 * 1 / (exp((X - μ) / s) + 1) ~ U(0, 1)
 */
template <typename RealType = double>
class RANDLIB_EXPORT LogisticRand : public randlib::ContinuousDistribution<RealType>
{
  double mu = 0;   ///< location μ
  double s = 1;    ///< scale s
  double logS = 0; ///< log(s)

public:
  LogisticRand(double location = 0, double scale = 1)
  {
    SetLocation(location);
    SetScale(scale);
  }

  String Name() const override
  {
    return "Logistic(" + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
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
      throw std::invalid_argument("Logistic distribution: scale of should be positive");
    s = scale;
    logS = std::log(s);
  }

  inline double GetLocation() const
  {
    return mu;
  }

  inline double GetScale() const
  {
    return s;
  }

  double f(const RealType& x) const override
  {
    double numerator = std::exp((mu - x) / s);
    double denominator = (1 + numerator);
    denominator *= denominator;
    denominator *= s;
    return numerator / denominator;
  }

  double logf(const RealType& x) const override
  {
    double x0 = (mu - x) / s;
    double y = RandMath::softplus(x0);
    y *= 2;
    y += logS;
    return x0 - y;
  }

  double F(const RealType& x) const override
  {
    double expX = std::exp((mu - x) / s);
    return 1.0 / (1 + expX);
  }

  double S(const RealType& x) const override
  {
    double expX = std::exp((mu - x) / s);
    return expX / (1 + expX);
  }

  RealType Variate() const override
  {
    /// there can be used rejection method from Laplace or Cauchy (Luc Devroye, p.
    /// 471) or ziggurat
    return mu + s * std::log(1.0 / UniformRand<RealType>::StandardVariate(this->localRandGenerator) - 1);
  }

  long double Mean() const override
  {
    return mu;
  }

  long double Variance() const override
  {
    double sPi = s * M_PI;
    return sPi * sPi / 3;
  }

  RealType Median() const override
  {
    return mu;
  }

  RealType Mode() const override
  {
    return mu;
  }

  long double Skewness() const override
  {
    return 0;
  }

  long double ExcessKurtosis() const override
  {
    return 1.2;
  }

  long double Entropy() const
  {
    return 2 + logS;
  }

  /**
   * @fn FitLocation
   * fit location parameter via maximum-likelihood
   * @param sample
   */
  void FitLocation(const std::vector<RealType>& sample)
  {
    double nHalf = 0.5 * sample.size();
    RealType root = 0;
    if(!RandMath::findRootNewtonFirstOrder<RealType>(
           [this, sample, nHalf](RealType m) {
             double f1 = 0, f2 = 0;
             for(const double& x : sample)
             {
               double aux = std::exp((m - x) / s);
               double denom = 1.0 + aux;
               f1 += 1.0 / denom;
               denom *= denom;
               f2 -= aux / denom;
             }
             f1 -= nHalf;
             return DoublePair(f1, f2);
           },
           root))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure"));
    SetLocation(root);
  }

private:
  RealType quantileImpl(double p) const override
  {
    return mu - s * (std::log1pl(-p) - std::log(p));
  }

  RealType quantileImpl1m(double p) const override
  {
    return mu - s * (std::log(p) - std::log1pl(-p));
  }

  std::complex<double> CFImpl(double t) const override
  {
    double pist = M_PI * s * t;
    std::complex<double> y(0.0, t * mu);
    y = std::exp(y);
    y *= pist;
    y /= std::sinh(pist);
    return y;
  }
};
} // namespace randlib
