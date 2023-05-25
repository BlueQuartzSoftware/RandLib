#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/CauchyRand.hpp"
#include "distributions/univariate/continuous/NakagamiRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"

namespace randlib
{
/**
 * @brief The StudentTRand class <BR>
 * Student's t-distribution
 *
 * Notation: X ~ t(ν, μ, σ)
 * If X ~ t(1, μ, σ), then X ~ Cauchy(μ, σ)
 * X -> Normal(μ, σ) for t -> ∞
 */
template <typename RealType = double>
class RANDLIB_EXPORT StudentTRand : public randlib::ContinuousDistribution<RealType>
{
  double nu = 1;       ///< degree ν
  double mu = 0;       ///< location μ
  double sigma = 1;    ///< scale σ
  double logSigma = 0; ///< log(σ)
  NakagamiRand<RealType> Y{};
  double pdfCoef = -M_LNPI;   ///< coefficient for faster pdf calculation
  double nup1Half = 1;        ///< 0.5 * (ν + 1)
  double logBetaFun = M_LNPI; ///< log(B(0.5 * ν, 0.5))

public:
  explicit StudentTRand(double degree = 1.0, double location = 0.0, double scale = 1.0)
  {
    SetDegree(degree);
    SetLocation(location);
    SetScale(scale);
  }

  String Name() const override
  {
    if(mu == 0.0 && sigma == 1.0)
      return "Student-t(" + this->toStringWithPrecision(GetDegree()) + ")";
    return "Student-t(" + this->toStringWithPrecision(GetDegree()) + ", " + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
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

  void SetDegree(double degree)
  {
    if(degree <= 0.0)
      throw std::invalid_argument("Student-t distribution: degree parameter should be positive");
    nu = degree;
    Y.SetParameters(0.5 * nu, 1.0);

    nup1Half = 0.5 * (nu + 1);
    pdfCoef = Y.GetLogGammaShapeRatio();
    pdfCoef -= 0.5 * M_LNPI;
    logBetaFun = -pdfCoef;
    pdfCoef -= 0.5 * std::log(nu);
  }

  void SetLocation(double location)
  {
    mu = location;
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Student-t distribution: scale should be positive");
    sigma = scale;
    logSigma = std::log(sigma);
  }

  inline double GetDegree() const
  {
    return nu;
  }

  inline double GetLocation() const
  {
    return mu;
  }

  inline double GetScale() const
  {
    return sigma;
  }

  double f(const RealType& x) const override
  {
    /// adjustment
    double x0 = x - mu;
    x0 /= sigma;
    double xSq = x0 * x0;
    if(nu == 1) /// Cauchy distribution
      return M_1_PI / (sigma * (1 + xSq));
    if(nu == 2)
    {
      double tmp = 2.0 + xSq;
      return 1.0 / (sigma * tmp * std::sqrt(tmp));
    }
    if(nu == 3)
    {
      double y = 3 + xSq;
      return 6 * M_SQRT3 * M_1_PI / (sigma * y * y);
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    /// adjustment
    double x0 = x - mu;
    x0 /= sigma;
    double xSq = x0 * x0;
    if(nu == 1) /// Cauchy distribution
      return -logSigma - M_LNPI - std::log1pl(xSq);
    if(nu == 2)
      return -logSigma - 1.5 * std::log(2.0 + xSq);
    if(nu == 3)
    {
      double y = -2 * std::log(3.0 + xSq);
      y += M_LN2 + 1.5 * M_LN3 - M_LNPI - logSigma;
      return y;
    }
    double y = -nup1Half * std::log1pl(xSq / nu);
    return pdfCoef + y - logSigma;
  }

  double F(const RealType& x) const override
  {
    double x0 = x - mu;
    x0 /= sigma;
    if(x0 == 0.0)
      return 0.5;
    if(nu == 1)
    {
      /// Cauchy distribution
      return 0.5 + M_1_PI * RandMath::atan(x0);
    }
    double xSq = x0 * x0;
    if(nu == 2)
    {
      return 0.5 + 0.5 * x0 / std::sqrt(2 + xSq);
    }
    if(nu == 3)
    {
      double y = M_SQRT3 * x0 / (xSq + 3);
      y += RandMath::atan(x0 / M_SQRT3);
      return 0.5 + M_1_PI * y;
    }
    double t = nu / (xSq + nu);
    double y = 0.5 * RandMath::ibeta(t, 0.5 * nu, 0.5, logBetaFun, std::log(t), std::log1pl(-t));
    return (x0 > 0.0) ? (1.0 - y) : y;
  }

  double S(const RealType& x) const override
  {
    double x0 = x - mu;
    x0 /= sigma;
    if(x0 == 0.0)
      return 0.5;
    if(nu == 1)
    {
      /// Cauchy distribution
      return 0.5 + M_1_PI * RandMath::atan(-x0);
    }
    double xSq = x0 * x0;
    if(nu == 2)
    {
      return 0.5 - 0.5 * x0 / std::sqrt(2 + xSq);
    }
    if(nu == 3)
    {
      double y = M_SQRT3 * x0 / (xSq + 3);
      y += RandMath::atan(x0 / M_SQRT3);
      return 0.5 - M_1_PI * y;
    }
    double t = nu / (xSq + nu);
    double y = 0.5 * RandMath::ibeta(t, 0.5 * nu, 0.5, logBetaFun, std::log(t), std::log1pl(-t));
    return (x0 > 0.0) ? y : 1.0 - y;
  }

  RealType Variate() const override
  {
    if(nu == 1)
      return mu + sigma * CauchyRand<RealType>::StandardVariate(this->localRandGenerator);
    return mu + sigma * randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator) / Y.Variate();
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    if(nu == 1)
    {
      for(RealType& var : outputData)
        var = mu + sigma * CauchyRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    else
    {
      Y.Sample(outputData);
      for(RealType& var : outputData)
        var = mu + sigma * randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator) / var;
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    Y.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return (nu > 1) ? mu : NAN;
  }

  long double Variance() const override
  {
    if(nu > 2)
      return sigma * sigma * nu / (nu - 2);
    return (nu > 1) ? INFINITY : NAN;
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
    return (nu > 3) ? 0.0 : NAN;
  }

  long double ExcessKurtosis() const override
  {
    if(nu > 4)
      return 6.0 / (nu - 4);
    return (nu > 2) ? INFINITY : NAN;
  }

private:
  RealType quantileImpl(double p) const override
  {
    double temp = p - 0.5;
    if(nu == 1)
      return std::tan(M_PI * temp) * sigma + mu;
    double pq = p * (1.0 - p);
    if(nu == 2)
      return sigma * 2.0 * temp * std::sqrt(0.5 / pq) + mu;
    if(nu == 4)
    {
      double alpha = 2 * std::sqrt(pq);
      double beta = std::cos(std::acos(alpha) / 3.0) / alpha - 1;
      return mu + sigma * 2 * RandMath::sign(temp) * std::sqrt(beta);
    }
    return randlib::ContinuousDistribution<RealType>::quantileImpl(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    double temp = 0.5 - p;
    if(nu == 1)
      return std::tan(M_PI * temp) * sigma + mu;
    double pq = p * (1.0 - p);
    if(nu == 2)
      return sigma * 2 * temp * std::sqrt(0.5 / pq) + mu;
    if(nu == 4)
    {
      double alpha = 2 * std::sqrt(pq);
      double beta = std::cos(std::acos(alpha) / 3.0) / alpha - 1;
      return mu + sigma * 2 * RandMath::sign(temp) * std::sqrt(beta);
    }
    return randlib::ContinuousDistribution<RealType>::quantileImpl1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    double x = std::sqrt(nu) * t * sigma; // value of sqrt(nu) can be hashed
    double vHalf = 0.5 * nu;
    double y = vHalf * std::log(x);
    y -= Y.GetLogGammaFunction();
    y -= (vHalf - 1) * M_LN2;
    y += RandMath::logBesselK(vHalf, x);
    double costmu = std::cos(t * mu), sintmu = std::sin(t * mu);
    std::complex<double> cf(costmu, sintmu);
    return std::exp(y) * cf;
  }
};
} // namespace randlib
