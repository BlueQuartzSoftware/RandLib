#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/StableRand.hpp"
#include "distributions/univariate/continuous/LaplaceRand.hpp"

namespace randlib
{
/**
 * @brief The GeneralGeometricStableDistribution class
 * Abstract class for general interface of Asymmetric Laplace
 * and Geometric-Stable distributions
 */
template <typename RealType = double>
class RANDLIB_EXPORT GeneralGeometricStableDistribution : public randlib::ContinuousDistribution<RealType>
{
  StableRand<RealType> Z{};

protected:
  double alpha = 2;              ///< characteristic exponent α
  double alphaInv = 0.5;         /// 1/α
  double beta = 0;               ///< skewness β
  double mu = 0;                 ///< location μ
  double m = 0;                  ///< shift m
  double gamma = M_SQRT2;        ///< scale γ
  double logGamma = 0.5 * M_LN2; ///< log(γ)

  /// parameters for α = 2
  double kappa = 1;            ///< asymmetry coefficient κ
  double kappaInv = 1;         ///< 1 / κ
  double kappaSq = 1;          ///< κ^2
  double log1pKappaSq = M_LN2; ///< log(1 + κ^2)
  double logKappa = 0;         /// log(κ)

  GeneralGeometricStableDistribution(double exponent, double skewness, double scale = 1.0, double location = 0.0, double shift = 0.0)
  {
    SetParameters(exponent, skewness, scale, location, shift);
  }

  virtual ~GeneralGeometricStableDistribution()
  {
  }

public:
  inline double GetExponent() const
  {
    return alpha;
  }

  inline double GetSkewness() const
  {
    return beta;
  }

  inline double GetScale() const
  {
    return gamma;
  }

  inline double GetLogScale() const
  {
    return logGamma;
  }

  SUPPORT_TYPE SupportType() const override
  {
    if(alpha < 1)
    {
      if(beta == 1 && mu >= 0)
        return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
      if(beta == -1 && mu <= 0)
        return SUPPORT_TYPE::LEFTSEMIFINITE_T;
    }
    return SUPPORT_TYPE::INFINITE_T;
  }

  RealType MinValue() const override
  {
    if(alpha < 1 && beta == 1 && mu >= 0)
      return m;
    return -INFINITY;
  }

  RealType MaxValue() const override
  {
    if(alpha < 1 && beta == -1 && mu <= 0)
      return m;
    return INFINITY;
  }

  double f(const RealType& x) const override
  {
    double x0 = x - m;
    /// Cut zeros for semi-infinite distribution
    if(alpha < 1 && std::fabs(beta) == 1)
    {
      if(x0 < 0 && mu >= 0 && beta > 0)
        return 0.0;
      if(x0 > 0 && mu <= 0 && beta < 0)
        return 0.0;
    }

    /// Verify if any of analytical expressions is applicable
    if(distributionType == LAPLACE || distributionType == ASYMMETRIC_LAPLACE)
      return pdfLaplace(x0);
    if(distributionType == CAUCHY)
      return pdfByCauchy(x0);
    if(distributionType == LEVY)
      return pdfByLevy(x0);

    if(x0 == 0)
    {
      if(alpha == 1)
        return INFINITY;
      if(mu == 0)
      {
        double coef = std::tgammal(1.0 - alphaInv);
        if(!std::isfinite(coef))
          return INFINITY;
        return Z.f(0) * coef / gamma;
      }
    }

    return RandMath::integral(
        [this, x0](double z) {
          if(z <= 0 || z >= 1)
            return 0.0;
          double logz = std::log(z);
          double tau = 1.0 / (gamma * std::pow(-logz, alphaInv));
          double par = tau * (x0 + mu * logz);
          return tau * Z.f(par);
        },
        0, 1);
  }

  double logf(const RealType& x) const override
  {
    return (distributionType == LAPLACE || distributionType == ASYMMETRIC_LAPLACE) ? logpdfLaplace(x - m) : std::log(f(x));
  }

  double F(const RealType& x) const override
  {
    double x0 = x - m;
    if(distributionType == LAPLACE || distributionType == ASYMMETRIC_LAPLACE)
      return cdfLaplace(x0);
    return RandMath::integral(
        [this, x0](double z) {
          if(z <= 0)
          {
            if(x0 != 0.0)
              return (x0 < 0.0) ? 0.0 : 1.0;
            if(mu == 0)
              return Z.F(0);
            if(distributionType == UNITY_EXPONENT)
              return Z.F(-mu / gamma);
            if(alpha < 1.0)
              return Z.F(0);
            return (mu < 0.0) ? 1.0 : 0.0;
          }

          if(z >= 1)
          {
            if(mu == 0.0)
              return Z.F(0);
            if(distributionType == UNITY_EXPONENT)
              return Z.F(-mu / gamma);
            if(alpha > 1.0)
              return Z.F(0);
            return (mu < 0.0) ? 1.0 : 0.0;
          }

          double denominator = 1.0 - z;
          double t = z / denominator;
          double par = x0 - mu * t;
          par /= std::pow(t, alphaInv) * gamma;
          double y = std::exp(-t) * Z.F(par);
          return y / (denominator * denominator);
        },
        0, 1);
  }

  RealType Variate() const override
  {
    switch(distributionType)
    {
    case LAPLACE:
      return gamma * randlib::LaplaceRand<RealType>::StandardVariate(this->localRandGenerator);
    case ASYMMETRIC_LAPLACE:
      return gamma * randlib::AsymmetricLaplaceRand<RealType>::StandardVariate(kappa, this->localRandGenerator);
    case ONEHALF_EXPONENT:
    case LEVY:
      return variateForOneHalfExponent(Z.Variate());
    case CAUCHY:
      return variateByCauchy(Z.Variate());
    case UNITY_EXPONENT:
      return variateForUnityExponent(Z.Variate());
    case GENERAL:
      return variateForGeneralExponent(Z.Variate());
    default:
      throw std::runtime_error("Shifted Geometric Stable distribution: invalid distribution type");
    }
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    switch(distributionType)
    {
    case LAPLACE: {
      for(RealType& var : outputData)
        var = gamma * randlib::LaplaceRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    break;
    case ASYMMETRIC_LAPLACE: {
      for(RealType& var : outputData)
        var = gamma * randlib::AsymmetricLaplaceRand<RealType>::StandardVariate(kappa, this->localRandGenerator);
    }
    break;
    case ONEHALF_EXPONENT:
    case LEVY: {
      Z.Sample(outputData);
      for(RealType& var : outputData)
        var = variateForOneHalfExponent(var);
    }
    break;
    case CAUCHY: {
      Z.Sample(outputData);
      for(RealType& var : outputData)
        var = variateByCauchy(var);
    }
    break;
    case UNITY_EXPONENT: {
      Z.Sample(outputData);
      for(RealType& var : outputData)
        var = variateForUnityExponent(var);
    }
    break;
    case GENERAL: {
      Z.Sample(outputData);
      for(RealType& var : outputData)
        var = variateForGeneralExponent(var);
    }
    break;
    default:
      throw std::runtime_error("Shifted Geometric Stable distribution: invalid distribution type");
    }
  }

  void Reseed(unsigned long seed) const override
  {
    Z.Reseed(seed);
  }

  long double Mean() const override
  {
    if(alpha > 1)
      return m + mu;
    if(beta == 1)
      return INFINITY;
    return (beta == -1) ? -INFINITY : NAN;
  }

  long double Variance() const override
  {
    if(distributionType == LAPLACE || distributionType == ASYMMETRIC_LAPLACE)
      return mu * mu + 2 * gamma * gamma;
    return INFINITY;
  }

  RealType Median() const override
  {
    if(distributionType == LAPLACE || distributionType == ASYMMETRIC_LAPLACE)
      return quantileLaplace(0.5);
    return randlib::ContinuousDistribution<RealType>::Median();
  }

  RealType Mode() const override
  {
    // TODO: calculate mode for more cases
    if(alpha == 1 || alpha == 2)
      return m;
    return randlib::ContinuousDistribution<RealType>::Mode();
  }

  long double Skewness() const override
  {
    if(distributionType != LAPLACE && distributionType != ASYMMETRIC_LAPLACE)
      return NAN;
    long double k4 = kappaSq * kappaSq;
    long double k6 = k4 * kappaSq;
    long double z = (k4 + 1);
    z *= std::sqrt(z);
    long double y = 2 * (1 - k6);
    return y / z;
  }

  long double ExcessKurtosis() const override
  {
    if(distributionType != LAPLACE && distributionType != ASYMMETRIC_LAPLACE)
      return NAN;
    long double denominator = kappaSq + kappaInv * kappaInv;
    denominator *= denominator;
    return 6.0 - 12.0 / denominator;
  }

protected:
  void SetParameters(double exponent, double skewness, double scale = 1.0, double location = 0.0, double shift = 0.)
  {
    if(exponent < 0.1 || exponent > 2.0)
      throw std::invalid_argument("Geometric-Stable distribution: exponent should be inside the interval "
                                  "[0.1, 2], but it's equal to " +
                                  std::to_string(exponent));
    if(std::fabs(skewness) > 1.0)
      throw std::invalid_argument("Geometric-Stable distribution: skewness should be inside the interval "
                                  "[-1, 1], but it's equal to " +
                                  std::to_string(skewness));

    alpha = exponent;
    alphaInv = 1.0 / alpha;
    beta = skewness;
    Z.SetExponent(alpha);
    Z.SetSkewness(beta);

    SetScale(scale);
    SetLocation(location);
    SetShift(shift);

    if(alpha == 2.0)
      distributionType = (mu == 0.0) ? LAPLACE : ASYMMETRIC_LAPLACE;
    else if(alpha == 0.5)
      distributionType = (std::fabs(beta) == 1.0) ? LEVY : ONEHALF_EXPONENT;
    else if(alpha == 1.0)
      distributionType = (beta == 0.0) ? CAUCHY : UNITY_EXPONENT;
    else
      distributionType = GENERAL;
  }

  void SetLocation(double location)
  {
    mu = location;
  }

  void SetShift(double shift)
  {
    m = shift;
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Asymmetric-Laplace distribution: scale should "
                                  "be positive, but it's equal to " +
                                  std::to_string(scale));
    gamma = scale;
    logGamma = std::log(gamma);
  }

  void SetAsymmetry(double asymmetry)
  {
    if(asymmetry <= 0.0)
      throw std::invalid_argument("Asymmetric-Laplace distribution: asymmetry parameter should be "
                                  "positive, but it's equal to " +
                                  std::to_string(asymmetry));
    kappa = asymmetry;
    kappaInv = 1.0 / kappa;
    kappaSq = kappa * kappa;
    log1pKappaSq = std::log1pl(kappaSq);
    logKappa = std::log(kappa);
  }

  enum DISTRIBUTION_TYPE
  {
    LAPLACE,            ///< α = 2, μ = 0
    ASYMMETRIC_LAPLACE, ///< α = 2, μ ≠ 0
    CAUCHY,             ///< α = 1, β = 0
    LEVY,               ///< α = 0.5, |β| = 1
    UNITY_EXPONENT,     ///< α = 1, β ≠ 0
    ONEHALF_EXPONENT,   ///< α = 0.5, |β| ≠ 1
    GENERAL             ///< the rest
  };

  DISTRIBUTION_TYPE distributionType = LAPLACE; ///< type of distribution (Laplace by default)

  double pdfLaplace(double x) const
  {
    return std::exp(logpdfLaplace(x));
  }

  double logpdfLaplace(double x) const
  {
    double y = x / gamma;
    y *= (x < 0) ? kappaInv : -kappa;
    return y - logGamma - log1pKappaSq + logKappa;
  }

  double cdfLaplace(double x) const
  {
    double y = x / gamma;
    if(x < 0)
    {
      y *= kappaInv;
      y += 2 * logKappa - log1pKappaSq;
      return std::exp(y);
    }
    return -std::expm1l(-log1pKappaSq - kappa * y);
  }

  double cdfLaplaceCompl(double x) const
  {
    double y = x / gamma;
    if(x < 0)
    {
      y *= kappaInv;
      y += 2 * logKappa - log1pKappaSq;
      return -std::expm1l(y);
    }
    return std::exp(-log1pKappaSq - kappa * y);
  }

  /**
   * @fn quantileLaplace
   * @param p input parameter in the interval (0, 1)
   * @return quantile for (Asymmetric) Laplace distribution
   */
  RealType quantileLaplace(double p) const
  {
    if(p < kappaSq / (1 + kappaSq))
    {
      RealType q = p / kappaSq + p;
      q = std::log(q);
      q *= kappa * gamma;
      return m + q;
    }
    else
    {
      RealType q = (kappaSq + 1) * (1 - p);
      q = std::log(q);
      q *= gamma / kappa;
      return m - q;
    }
  }

  /**
   * @fn quantileLaplace1m
   * @param p input parameter in the interval (0, 1)
   * @return quantile of 1-p for (Asymmetric) Laplace distribution
   */
  RealType quantileLaplace1m(double p) const
  {
    if(p > 1.0 / (1 + kappaSq))
    {
      RealType pm1 = p - 1.0;
      RealType q = -pm1 / kappaSq - pm1;
      q = std::log(q);
      q *= kappa * gamma;
      return m + q;
    }
    else
    {
      RealType q = (kappaSq + 1) * p;
      q = std::log(q);
      q *= gamma / kappa;
      return m - q;
    }
  }

  std::complex<double> CFImpl(double t) const override
  {
    double x = 0;
    if(alpha != 2 && beta != 0)
    {
      x = (alpha == 1) ? M_2_PI * std::log(t) : -std::tan(M_PI_2 * alpha);
      x *= beta;
    }
    double re = std::pow(gamma * t, alpha);
    std::complex<double> psi = std::complex<double>(re, re * x - mu * t);
    return 1.0 / (1.0 + psi);
  }

private:
  double pdfByLevy(double x) const
  {
    if(mu == 0)
    {
      /// Invert parameter for case of -Levy
      x *= beta;
      if(x < 0)
        return 0.0;
      if(x == 0)
        return INFINITY;
      double halfSigmaInv = 0.5 / gamma;
      double z = std::sqrt(halfSigmaInv * x);
      double y = M_1_SQRTPI - RandMath::xexpxsqerfc(z);
      return 0.5 * y / (gamma * z);
    }

    if(x == 0)
    {
      return std::fabs(1.0 / mu);
    }

    double a = x / gamma, c = mu / gamma;
    if(beta < 0)
    {
      a = -a;
      c = -c;
    }

    if(a < 0)
    {
      if(c >= 0.0)
        return 0.0;
      double h = std::sqrt(1 - 2 * c); // can be hashed
      double hm1 = h - 1;
      double y = 2 * a / (hm1 * hm1);
      y -= std::log(gamma * hm1 * h);
      y += M_LN2;
      return std::exp(y);
    }

    double sqrt2a = std::sqrt(2 * a);

    if(c < 0.5)
    {
      double h = std::sqrt(1 - 2 * c); // can be hashed
      double term1 = RandMath::sign(c) * sqrt2a / (1 - h);
      double term2 = sqrt2a / (1 + h);
      double y = RandMath::xexpxsqerfc(term1);
      y -= RandMath::xexpxsqerfc(term2);
      y /= h * gamma * sqrt2a;
      return y;
    }

    if(c == 0.5)
    {
      double y = RandMath::xexpxsqerfc(sqrt2a);
      y *= (8.0 * a + 2.0) / sqrt2a;
      y -= 4 * sqrt2a / M_SQRTPI;
      return y / gamma;
    }

    /// we do numerical integration in the case of c > 0.5
    // (as I wasn't able to handle complex integrals)
    double adivc = x / mu;
    double y = RandMath::integral(
        [this, a, c, adivc](double t) {
          if(t <= 0 || t >= adivc)
            return 0.0;
          double amct = a - c * t;
          double integrand = 0.5 * t / amct;
          ++integrand;
          integrand *= t;
          integrand = std::exp(-integrand);
          integrand *= t;
          integrand /= amct * std::sqrt(amct);
          return integrand;
        },
        0, adivc);
    return y / (gamma * M_SQRT2PI);
  }

  double pdfByCauchy(double x) const
  {
    // TODO: find analytical solution, maybe using residue technique
    // otherwise, use peak, which is t = exp(x / sqrt(mu^2 + sigma^2))
    // to split the integral on two
    if(x == 0)
      return INFINITY;
    return -gamma * M_1_PI *
           RandMath::integral(
               [this, x](double t) {
                 if(t <= 0.0 || t >= 1.0)
                   return 0.0;
                 double logt = std::log(t);
                 double a = gamma * logt;
                 a *= a;
                 double b = x + mu * logt;
                 b *= b;
                 return logt / (a + b);
               },
               0, 1);
  }

  double variateForUnityExponent(double z) const
  {
    double W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    double X = z + 2 * beta * std::log(gamma * W) / M_PI;
    X *= gamma;
    X += mu;
    return X * W;
  }

  double variateForGeneralExponent(double z) const
  {
    double W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    double X = std::pow(W, alphaInv) * gamma * z;
    return mu * W + X;
  }

  double variateForOneHalfExponent(double z) const
  {
    double W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    double X = mu + gamma * W * z;
    return X * W;
  }

  double variateByCauchy(double z) const
  {
    double W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    double X = mu + gamma * z;
    return X * W;
  }
};

/**
 * @brief The GeometricStableRand class <BR>
 * Geometric-Stable distribution
 *
 * Notation: X ~ GS(α, β, γ, μ)
 *
 * If X ~ Laplace(m, γ, κ), then X - m ~ GS(2, β, γ, (1/κ - κ) * γ) with
 * arbitrary β
 */
template <typename RealType = double>
class RANDLIB_EXPORT GeometricStableRand : public GeneralGeometricStableDistribution<RealType>
{
public:
  GeometricStableRand(double exponent = 2, double skewness = 0, double scale = M_SQRT2, double location = 0)
  {
    ChangeAsymmetry();
  }

  virtual ~GeometricStableRand() = default;

  String Name() const override
  {
    return "Geometric Stable(" + this->toStringWithPrecision(this->GetExponent()) + ", " + this->toStringWithPrecision(this->GetSkewness()) + ", " + this->toStringWithPrecision(this->GetScale()) +
           ", " + this->toStringWithPrecision(this->GetLocation()) + ")";
  }

  void SetParameters(double exponent, double skewness)
  {
    GeneralGeometricStableDistribution<RealType>::SetParameters(exponent, skewness);
    ChangeAsymmetry();
  }

  void SetLocation(double location)
  {
    GeneralGeometricStableDistribution<RealType>::SetLocation(location);
    ChangeAsymmetry();
  }

  void SetScale(double scale)
  {
    GeneralGeometricStableDistribution<RealType>::SetScale(scale);
    ChangeAsymmetry();
  }

  inline double GetLocation() const
  {
    return this->mu;
  }

private:
  void ChangeAsymmetry()
  {
    if(this->distributionType != this->LAPLACE && this->distributionType != this->ASYMMETRIC_LAPLACE)
      return;
    double gamma2 = 2 * this->gamma;
    double asymmetry = this->mu * this->mu + gamma2 * gamma2;
    asymmetry = std::sqrt(asymmetry) - this->mu;
    asymmetry /= gamma2;
    this->SetAsymmetry(asymmetry);
  }
};
} // namespace randlib
