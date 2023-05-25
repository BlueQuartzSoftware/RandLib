#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/CauchyRand.hpp"
#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/LevyRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

#include <algorithm>
#include <functional>

namespace randlib
{
/**
 * @brief The StableDistribution class <BR>
 * Abstract class for Stable distribution
 *
 * Notation: X ~ S(α, β, γ, μ)
 *
 * Related distributions: <BR>
 * If X ~ Normal(μ, σ), then X ~ S(2, 0, σ/√2, μ) <BR>
 * If X ~ Cauchy(μ, γ), then X ~ S(1, 0, γ, μ) <BR>
 * If +/-X ~ Levy(μ, γ), then X ~ S(0.5, +/-1, γ, μ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT StableDistribution : virtual public randlib::ContinuousDistribution<RealType>
{
protected:
  double alpha = 2;              ///< characteristic exponent α
  double beta = 0;               ///< skewness β
  double mu = 0;                 ///< location μ
  double gamma = M_SQRT2;        ///< scale γ
  double logGamma = 0.5 * M_LN2; ///< log(γ)

  double pdfCoef = 0.5 * (M_LN2 + M_LNPI); ///< hashed coefficient for faster pdf calculations
  double pdftailBound = INFINITY;          ///< boundary k such that for |x| > k we can
  ///< use pdf tail approximation
  double cdftailBound = INFINITY; ///< boundary k such that for |x| > k we can
  ///< use cdf tail approximation

  StableDistribution(double exponent, double skewness, double scale = 1, double location = 0)
  {
    SetParameters(exponent, skewness, scale, location);
  }

  virtual ~StableDistribution() = default;

public:
  SUPPORT_TYPE SupportType() const override
  {
    if(alpha < 1)
    {
      if(beta == 1)
        return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
      if(beta == -1)
        return SUPPORT_TYPE::LEFTSEMIFINITE_T;
    }
    return SUPPORT_TYPE::INFINITE_T;
  }

  RealType MinValue() const override
  {
    return (alpha < 1 && beta == 1) ? mu : -INFINITY;
  }

  RealType MaxValue() const override
  {
    return (alpha < 1 && beta == -1) ? mu : INFINITY;
  }

  void SetLocation(double location)
  {
    SetParameters(alpha, beta, gamma, location);
  }

  void SetScale(double scale)
  {
    SetParameters(alpha, beta, scale, mu);
  }

  /**
   * @fn GetExponent
   * @return characteristic exponent α
   */
  inline double GetExponent() const
  {
    return alpha;
  }

  /**
   * @fn GetSkewness
   * @return skewness parameter β
   */
  inline double GetSkewness() const
  {
    return beta;
  }

  /**
   * @fn GetScale
   * @return scale parameter γ
   */
  inline double GetScale() const
  {
    return gamma;
  }

  /**
   * @fn GetLocation
   * @return location parameter μ
   */
  inline double GetLocation() const
  {
    return mu;
  }

  /**
   * @fn GetLogScale
   * @return logarithm of the scale parameter γ
   */
  inline double GetLogScale() const
  {
    return logGamma;
  }

  double f(const RealType& x) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return pdfNormal(x);
    case CAUCHY:
      return pdfCauchy(x);
    case LEVY:
      return (beta > 0) ? pdfLevy(x) : pdfLevy(2 * mu - x);
    case UNITY_EXPONENT:
      return pdfForUnityExponent(x);
    case GENERAL:
      return pdfForGeneralExponent(x);
    default:
      throw std::runtime_error("Stable distribution: invalid distribution type");
    }
  }

  double logf(const RealType& x) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return logpdfNormal(x);
    case CAUCHY:
      return logpdfCauchy(x);
    case LEVY:
      return (beta > 0) ? logpdfLevy(x) : logpdfLevy(2 * mu - x);
    case UNITY_EXPONENT:
      return std::log(pdfForUnityExponent(x));
    case GENERAL:
      return std::log(pdfForGeneralExponent(x));
    default:
      throw std::runtime_error("Stable distribution: invalid distribution type");
    }
  }

  double F(const RealType& x) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return cdfNormal(x);
    case CAUCHY:
      return cdfCauchy(x);
    case LEVY:
      return (beta > 0) ? cdfLevy(x) : cdfLevyCompl(2 * mu - x);
    case UNITY_EXPONENT:
      return cdfForUnityExponent(x);
    case GENERAL:
      return cdfForGeneralExponent(x);
    default:
      throw std::runtime_error("Stable distribution: invalid distribution type");
    }
  }

  double S(const RealType& x) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return cdfNormalCompl(x);
    case CAUCHY:
      return cdfCauchyCompl(x);
    case LEVY:
      return (beta > 0) ? cdfLevyCompl(x) : cdfLevy(2 * mu - x);
    case UNITY_EXPONENT:
      return 1.0 - cdfForUnityExponent(x);
    case GENERAL:
      return 1.0 - cdfForGeneralExponent(x);
    default:
      throw std::runtime_error("Stable distribution: invalid distribution type");
    }
  }

  RealType Variate() const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return mu + M_SQRT2 * gamma * NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    case CAUCHY:
      return mu + gamma * CauchyRand<RealType>::StandardVariate(this->localRandGenerator);
    case LEVY:
      return mu + RandMath::sign(beta) * gamma * LevyRand<RealType>::StandardVariate(this->localRandGenerator);
    case UNITY_EXPONENT:
      return variateForUnityExponent();
    case GENERAL:
      return (alpha == 0.5) ? variateForExponentEqualOneHalf() : variateForGeneralExponent();
    default:
      throw std::runtime_error("Stable distribution: invalid distribution type");
    }
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    switch(distributionType)
    {
    case NORMAL: {
      double stdev = M_SQRT2 * gamma;
      for(RealType& var : outputData)
        var = mu + stdev * NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    break;
    case CAUCHY: {
      for(RealType& var : outputData)
        var = mu + gamma * CauchyRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    break;
    case LEVY: {
      if(beta > 0)
      {
        for(RealType& var : outputData)
          var = mu + gamma * LevyRand<RealType>::StandardVariate(this->localRandGenerator);
      }
      else
      {
        for(RealType& var : outputData)
          var = mu - gamma * LevyRand<RealType>::StandardVariate(this->localRandGenerator);
      }
    }
    break;
    case UNITY_EXPONENT: {
      for(RealType& var : outputData)
        var = variateForUnityExponent();
    }
    break;
    case GENERAL: {
      if(alpha == 0.5)
      {
        for(RealType& var : outputData)
          var = variateForExponentEqualOneHalf();
      }
      else
      {
        for(RealType& var : outputData)
          var = variateForGeneralExponent();
      }
    }
    break;
    default:
      break;
    }
  }

  long double Mean() const override
  {
    if(alpha > 1)
      return mu;
    if(beta == 1)
      return INFINITY;
    return (beta == -1) ? -INFINITY : NAN;
  }

  long double Variance() const override
  {
    return (distributionType == NORMAL) ? 2 * gamma * gamma : INFINITY;
  }

  RealType Median() const override
  {
    /// For symmetric and normal distributions mode is μ
    if(beta == 0 || distributionType == NORMAL)
      return mu;
    return randlib::ContinuousDistribution<RealType>::Median();
  }

  RealType Mode() const override
  {
    /// For symmetric and normal distributions mode is μ
    if(beta == 0 || distributionType == NORMAL)
      return mu;
    if(distributionType == LEVY)
      return mu + beta * gamma / 3.0;
    return randlib::ContinuousDistribution<RealType>::Mode();
  }

  long double Skewness() const override
  {
    return (distributionType == NORMAL) ? 0 : NAN;
  }

  long double ExcessKurtosis() const override
  {
    return (distributionType == NORMAL) ? 0 : NAN;
  }

protected:
  void SetParameters(double exponent, double skewness, double scale = 1, double location = 0)
  {
    parametersVerification(exponent, skewness, scale);
    alpha = exponent;
    alphaInv = 1.0 / alpha;
    beta = skewness;
    mu = location;
    gamma = scale;
    logGamma = std::log(gamma);
    alpha_alpham1 = alpha / (alpha - 1.0);

    /// Set id of distribution
    if(alpha == 2.0)
      return setParametersForNormal();
    if(alpha == 1.0)
      return (beta == 0.0) ? setParametersForCauchy() : setParametersForUnityExponent();
    if(alpha == 0.5 && std::fabs(beta) == 1.0)
      return setParametersForLevy();
    return setParametersForGeneralExponent();
  }

  /**
   * @fn pdfNormal
   * @param x
   * @return probability density function of normal distribution
   */
  double pdfNormal(RealType x) const
  {
    return std::exp(logpdfNormal(x));
  }

  /**
   * @fn logpdfNormal
   * @param x
   * @return logarithm of probability density function of normal distribution
   */
  double logpdfNormal(RealType x) const
  {
    double y = x - mu;
    y *= 0.5 / gamma;
    y *= y;
    y += pdfCoef;
    return -y;
  }

  /**
   * @fn pdfCauchy
   * @param x
   * @return probability density function of Cauchy distribution
   */
  double pdfCauchy(RealType x) const
  {
    double y = x - mu;
    y *= y;
    y /= gamma;
    y += gamma;
    return M_1_PI / y;
  }

  /**
   * @fn logpdfCauchy
   * @param x
   * @return logarithm of probability density function of Cauchy distribution
   */
  double logpdfCauchy(RealType x) const
  {
    double x0 = x - mu;
    x0 /= gamma;
    double xSq = x0 * x0;
    return pdfCoef - std::log1pl(xSq);
  }

  /**
   * @fn pdfLevy
   * @param x
   * @return probability density function of Levy distribution
   */
  double pdfLevy(RealType x) const
  {
    return (x <= mu) ? 0.0 : std::exp(logpdfLevy(x));
  }

  /**
   * @fn logpdfLevy
   * @param x
   * @return logarithm of probability density function of Levy distribution
   */
  double logpdfLevy(RealType x) const
  {
    double x0 = x - mu;
    if(x0 <= 0.0)
      return -INFINITY;
    double y = gamma / x0;
    y += 3 * std::log(x0);
    y -= pdfCoef;
    return -0.5 * y;
  }

  /**
   * @fn cdfNormal
   * @param x
   * @return cumulative distribution function of normal distribution
   */
  double cdfNormal(RealType x) const
  {
    double y = mu - x;
    y *= 0.5 / gamma;
    return 0.5 * std::erfc(y);
  }

  /**
   * @fn cdfNormalCompl
   * @param x
   * @return complementary cumulative distribution function of normal
   * distribution
   */
  double cdfNormalCompl(RealType x) const
  {
    double y = x - mu;
    y *= 0.5 / gamma;
    return 0.5 * std::erfc(y);
  }

  /**
   * @fn cdfCauchy
   * @param x
   * @return cumulative distribution function of Cauchy distribution
   */
  double cdfCauchy(RealType x) const
  {
    double x0 = x - mu;
    x0 /= gamma;
    return 0.5 + M_1_PI * RandMath::atan(x0);
  }

  /**
   * @fn cdfCauchyCompl
   * @param x
   * @return complementary cumulative distribution function of Cauchy
   * distribution
   */
  double cdfCauchyCompl(RealType x) const
  {
    double x0 = mu - x;
    x0 /= gamma;
    return 0.5 + M_1_PI * RandMath::atan(x0);
  }

  /**
   * @fn cdfLevy
   * @param x
   * @return cumulative distribution function of Levy distribution
   */
  double cdfLevy(RealType x) const
  {
    if(x <= mu)
      return 0;
    double y = x - mu;
    y += y;
    y = gamma / y;
    y = std::sqrt(y);
    return std::erfc(y);
  }

  /**
   * @fn cdfLevyCompl
   * @param x
   * @return complementary cumulative distribution function of Levy distribution
   */
  double cdfLevyCompl(RealType x) const
  {
    if(x <= mu)
      return 1.0;
    double y = x - mu;
    y += y;
    y = gamma / y;
    y = std::sqrt(y);
    return std::erf(y);
  }

  /**
   * @fn quantileNormal
   * @param p input parameter in the interval (0, 1)
   * @return quantile for Gaussian distribution
   */
  RealType quantileNormal(double p) const
  {
    return mu - 2 * gamma * RandMath::erfcinv(2 * p);
  }

  /**
   * @fn quantileNormal1m
   * @param p input parameter in the interval (0, 1)
   * @return quantile of 1-p for Gaussian distribution
   */
  RealType quantileNormal1m(double p) const
  {
    return mu + 2 * gamma * RandMath::erfcinv(2 * p);
  }

  /**
   * @fn quantileCauchy
   * @param p input parameter in the interval (0, 1)
   * @return quantile for Cauchy distribution
   */
  RealType quantileCauchy(double p) const
  {
    return mu - gamma / std::tan(M_PI * p);
  }

  /**
   * @fn quantileCauchy1m
   * @param p input parameter in the interval (0, 1)
   * @return quantile of 1-p for Cauchy distribution
   */
  RealType quantileCauchy1m(double p) const
  {
    return mu + gamma / std::tan(M_PI * p);
  }

  /**
   * @fn quantileLevy
   * @param p input parameter in the interval (0, 1)
   * @return quantile for Levy distribution
   */
  RealType quantileLevy(double p) const
  {
    double y = RandMath::erfcinv(p);
    return mu + 0.5 * gamma / (y * y);
  }

  /**
   * @fn quantileLevy1m
   * @param p input parameter in the interval (0, 1)
   * @return quantile of 1-p for Levy distribution
   */
  RealType quantileLevy1m(double p) const
  {
    double y = RandMath::erfinv(p);
    return mu + 0.5 * gamma / (y * y);
  }

  /**
   * @fn cfNormal
   * @param t positive parameter
   * @return characteristic function for Gaussian distribution
   */
  std::complex<double> cfNormal(double t) const
  {
    double gammaT = gamma * t;
    std::complex<double> y(-gammaT * gammaT, mu * t);
    return std::exp(y);
  }

  /**
   * @fn cfCauchy
   * @param t positive parameter
   * @return characteristic function for Cauchy distribution
   */
  std::complex<double> cfCauchy(double t) const
  {
    std::complex<double> y(-gamma * t, mu * t);
    return std::exp(y);
  }

  /**
   * @fn cfLevy
   * @param t positive parameter
   * @return characteristic function for Levy distribution
   */
  std::complex<double> cfLevy(double t) const
  {
    std::complex<double> y(0.0, -2 * gamma * t);
    y = -std::sqrt(y);
    y += std::complex<double>(0.0, mu * t);
    return std::exp(y);
  }

private:
  double alphaInv = 0.5;                      ///< 1/α
  double zeta = 0;                            ///< ζ = -β * tan(πα/2)
  double omega = 0;                           ///< ω = log(1 + ζ^2) / (2α)
  double xi = 0;                              ///< ξ = atan(-ζ) / α;
  double alpha_alpham1 = 2;                   ///< α / (α - 1)
  double logGammaPi_2 = M_LNPI - 1.5 * M_LN2; ///< log(γπ/2)

  static constexpr double BIG_NUMBER = 1e9;     ///< a.k.a. infinity for pdf and cdf calculations
  static constexpr double ALMOST_TWO = 1.99999; ///< parameter used to identify α close to 2

  enum DISTRIBUTION_TYPE
  {
    NORMAL,         ///< α = 2
    LEVY,           ///< α = 0.5, |β| = 1
    CAUCHY,         ///< α = 1, β = 0
    UNITY_EXPONENT, ///< α = 1, β ≠ 0
    GENERAL         ///< the rest
  };

  DISTRIBUTION_TYPE distributionType = NORMAL; ///< type of distribution (Gaussian by default)

  void parametersVerification(double exponent, double skewness, double scale)
  {
    if(exponent < 0.1 || exponent > 2.0)
      throw std::invalid_argument("Stable distribution: exponent should be in "
                                  "the interval [0.1, 2], but it's equal to " +
                                  std::to_string(exponent));
    if(std::fabs(skewness) > 1.0)
      throw std::invalid_argument("Stable distribution: skewness should be in "
                                  "the interval [-1, 1], but it's equal to " +
                                  std::to_string(skewness));
    if(scale <= 0.0)
      throw std::invalid_argument("Stable distribution: scale should be positive, but it's equal to " + std::to_string(scale));

    /// the following errors should be removed in the future
    if(exponent != 1.0 && std::fabs(exponent - 1.0) < 0.01 && skewness != 0.0)
      throw std::invalid_argument("Stable distribution: exponent close to 1 with "
                                  "non-zero skewness is not yet supported");
    if(exponent == 1.0 && skewness != 0.0 && std::fabs(skewness) < 0.01)
      throw std::invalid_argument("Stable distribution: skewness close to 0 with "
                                  "exponent equal to 1 is not yet supported");
  }

  void setParametersForNormal()
  {
    distributionType = NORMAL;
    pdfCoef = M_LN2 + logGamma + 0.5 * M_LNPI;
  }

  void setParametersForCauchy()
  {
    distributionType = CAUCHY;
    pdfCoef = -logGamma - M_LNPI;
  }

  void setParametersForLevy()
  {
    distributionType = LEVY;
    pdfCoef = logGamma - M_LN2 - M_LNPI;
  }

  void setParametersForUnityExponent()
  {
    distributionType = UNITY_EXPONENT;
    pdfCoef = 0.5 / (gamma * std::fabs(beta));
    pdftailBound = 0; // not in the use for now
    logGammaPi_2 = logGamma + M_LNPI - M_LN2;
  }

  void setParametersForGeneralExponent()
  {
    distributionType = GENERAL;
    if(beta != 0.0)
    {
      zeta = -beta * std::tan(M_PI_2 * alpha);
      omega = 0.5 * alphaInv * std::log1pl(zeta * zeta);
      xi = alphaInv * RandMath::atan(-zeta);
    }
    else
    {
      zeta = omega = xi = 0.0;
    }
    pdfCoef = M_1_PI * std::fabs(alpha_alpham1) / gamma;
    pdftailBound = 3.0 / (1.0 + alpha) * M_LN10;
    cdftailBound = 3.0 / alpha * M_LN10;
    /// define boundaries of region near 0, where we use series expansion
    if(alpha <= ALMOST_TWO)
    {
      seriesZeroParams.first = std::round(std::min(alpha * alpha * 40 + 1, 10.0));
      seriesZeroParams.second = -(alphaInv * 1.5 + 0.5) * M_LN10; /// corresponds to interval [10^(-15.5), ~0.056]
    }
    else
    {
      seriesZeroParams.first = 85;
      seriesZeroParams.second = M_LN2 + M_LN3; ///< corresponds to 6
    }
  }

  /**
   * @fn fastpdfExponentiation
   * @param u
   * @return exp(u - exp(u)), accelerated by truncation of input u
   */
  static double fastpdfExponentiation(double u)
  {
    if(u > 5 || u < -50)
      return 0.0;
    return (u < -25) ? std::exp(u) : std::exp(u - std::exp(u));
  }

  /**
   * @fn pdfShortTailExpansionForUnityExponent
   * @param logX
   * @return leading term of pdf short tail series expansion for large x, |β| =
   * 1 and α = 1
   */
  double pdfShortTailExpansionForUnityExponent(double x) const
  {
    if(x > 10)
      return 0.0;
    double xm1 = x - 1.0;
    double y = 0.5 * xm1 - std::exp(xm1);
    y -= 0.5 * (M_LN2 + M_LNPI);
    return std::exp(y);
  }

  /**
   * @fn limitCaseForIntegrandAuxForUnityExponent
   * @param theta
   * @param xAdj
   * @return large values in the case of closeness to extreme points
   */
  double limitCaseForIntegrandAuxForUnityExponent(double theta, double xAdj) const
  {
    if(theta > 0.0)
    {
      if(beta > 0.0)
        return BIG_NUMBER;
      return (beta == -1) ? xAdj - 1.0 : -BIG_NUMBER;
    }
    if(beta < 0.0)
      return BIG_NUMBER;
    return (beta == 1) ? xAdj - 1.0 : -BIG_NUMBER;
  }

  /**
   * @fn integrandAuxForUnityExponent
   * @param theta
   * @param xAdj
   * @return supplementary value for the integrand used in pdf calculations for
   * α = 1
   */
  double integrandAuxForUnityExponent(double theta, double xAdj) const
  {
    if(std::fabs(theta) >= M_PI_2)
      return limitCaseForIntegrandAuxForUnityExponent(theta, xAdj);
    if(theta == 0.0)
      return xAdj + M_LNPI - M_LN2;
    double thetaAdj = (M_PI_2 + beta * theta) / std::cos(theta);
    double u = std::log(thetaAdj);
    u += thetaAdj * std::sin(theta) / beta;
    return std::isfinite(u) ? u + xAdj : limitCaseForIntegrandAuxForUnityExponent(theta, xAdj);
  }

  /**
   * @fn integrandForUnityExponent
   * @param theta
   * @param xAdj
   * @return the value of the integrand used for calculations of pdf for α = 1
   */
  double integrandForUnityExponent(double theta, double xAdj) const
  {
    if(std::fabs(theta) >= M_PI_2)
      return 0.0;
    double u = integrandAuxForUnityExponent(theta, xAdj);
    return fastpdfExponentiation(u);
  }

  /**
   * @fn pdfForUnityExponent
   * @param x
   * @return value of probability density function for α = 1
   */
  double pdfForUnityExponent(double x) const
  {
    double xSt = (x - mu) / gamma;
    double xAdj = -M_PI_2 * xSt / beta - logGammaPi_2;

    /// We squeeze boudaries for extremely sharp integrands
    double boundary = RandMath::atan(M_2_PI * beta * (5.0 - xAdj));
    double upperBoundary = (beta > 0.0) ? boundary : M_PI_2;
    double lowerBoundary = (beta < 0.0) ? boundary : -M_PI_2;

    /// Find peak of the integrand
    double theta0 = 0;
    std::function<double(double)> funPtr = std::bind(&StableDistribution<RealType>::integrandAuxForUnityExponent, this, std::placeholders::_1, xAdj);
    RandMath::findRootBrentFirstOrder(funPtr, lowerBoundary, upperBoundary, theta0);

    /// Sanity check
    /// if we failed to find the peak position
    /// we set it in the middle between boundaries
    if(theta0 >= upperBoundary || theta0 <= lowerBoundary)
      theta0 = 0.5 * (upperBoundary + lowerBoundary);

    std::function<double(double)> integrandPtr = std::bind(&StableDistribution<RealType>::integrandForUnityExponent, this, std::placeholders::_1, xAdj);

    /// If theta0 is too close to +/-π/2, we can still underestimate the value of
    /// integral
    int maxRecursionDepth = 11;
    double closeness = M_PI_2 - std::fabs(theta0);
    if(closeness < 0.1)
      maxRecursionDepth = 20;
    else if(closeness < 0.2)
      maxRecursionDepth = 15;

    double int1 = RandMath::integral(integrandPtr, lowerBoundary, theta0, 1e-11, maxRecursionDepth);
    double int2 = RandMath::integral(integrandPtr, theta0, upperBoundary, 1e-11, maxRecursionDepth);
    return pdfCoef * (int1 + int2);
  }

  DoublePair seriesZeroParams{};

  /**
   * @fn pdfShortTailExpansionForGeneralExponent
   * @param logX
   * @return leading term of pdf short tail series expansion for |β| = 1 and
   * [(large x and α > 1) or (small x and α < 1)]
   */
  double pdfShortTailExpansionForGeneralExponent(double logX) const
  {
    double logAlpha = std::log(alpha);
    double log1mAlpha = (alpha < 1) ? std::log1pl(-alpha) : std::log(alpha - 1);
    double temp = logX - logAlpha;
    double y = std::exp(alpha_alpham1 * temp);
    y *= -std::fabs(1.0 - alpha);
    double z = (1.0 - 0.5 * alpha) / (alpha - 1) * temp;
    z -= 0.5 * (M_LN2 + M_LNPI + logAlpha + log1mAlpha);
    return std::exp(y + z);
  }

  /**
   * @fn pdfAtZero
   * @return probability density function for x = 0
   */
  double pdfAtZero() const
  {
    double y0 = 0.0;
    if(beta == 0.0)
      y0 = std::tgammal(alphaInv);
    else
    {
      y0 = std::lgammal(alphaInv) - omega;
      y0 = std::exp(y0) * std::cos(xi);
    }
    return y0 * M_1_PI / alpha;
  }

  /**
   * @fn pdfSeriesExpansionAtZero
   * @param logX log(x)
   * @param xiAdj adjusted ξ
   * @param k number of elements in series
   * @return series expansion of probability density function for x near 0
   */
  double pdfSeriesExpansionAtZero(double logX, double xiAdj, int k) const
  {
    /// Calculate first term of the sum
    /// (if x is 0, only this term is non-zero)
    double y0 = pdfAtZero();
    double sum = 0.0;
    if(beta == 0.0)
    {
      /// Symmetric distribution
      for(int n = 1; n <= k; ++n)
      {
        int n2 = n + n;
        double term = std::lgammal((n2 + 1) / alpha);
        term += n2 * logX;
        term -= RandMath::lfact(n2);
        term = std::exp(term);
        sum += (n & 1) ? -term : term;
      }
    }
    else
    {
      /// Asymmetric distribution
      double rhoPi_alpha = M_PI_2 + xiAdj;
      for(int n = 1; n <= k; ++n)
      {
        int np1 = n + 1;
        double term = std::lgammal(np1 * alphaInv);
        term += n * logX;
        term -= RandMath::lfact(n);
        term = std::exp(term - omega);
        term *= std::sin(np1 * rhoPi_alpha);
        sum += (n & 1) ? -term : term;
      }
    }
    return y0 + sum * M_1_PI / alpha;
  }

  /**
   * @fn pdfSeriesExpansionAtInf
   * @param logX log(x)
   * @param xiAdj adjusted ξ
   * @return series expansion of probability density function for large x
   */
  double pdfSeriesExpansionAtInf(double logX, double xiAdj) const
  {
    static constexpr int k = 10; ///< number of elements in the series
    double rhoPi = M_PI_2 + xiAdj;
    rhoPi *= alpha;
    double sum = 0.0;
    for(int n = 1; n <= k; ++n)
    {
      double aux = n * alpha + 1.0;
      double term = std::lgammal(aux);
      term -= aux * logX;
      term -= RandMath::lfact(n);
      term = std::exp(term - omega);
      term *= std::sin(rhoPi * n);
      sum += (n & 1) ? term : -term;
    }
    return M_1_PI * sum;
  }

  /**
   * @fn pdfTaylorExpansionTailNearCauchy
   * @param x
   * @return the Taylor approximated difference ~ f(x, α) - f(x, 1)
   */
  double pdfTaylorExpansionTailNearCauchy(double x) const
  {
    double xSq = x * x;
    double y = 1.0 + xSq;
    double ySq = y * y;
    double z = RandMath::atan(x);
    double zSq = z * z;
    double logY = std::log1pl(xSq);
    double alpham1 = alpha - 1.0;
    double temp = 1.0 - M_EULER - 0.5 * logY;
    /// first derivative
    double f_a = temp;
    f_a *= xSq - 1.0;
    f_a += 2 * x * z;
    f_a /= ySq;
    static constexpr long double M_PI_SQ_6 = 1.64493406684822643647l; /// π^2 / 6
    /// second derivative
    double f_aa1 = M_PI_SQ_6;
    f_aa1 += temp * temp;
    f_aa1 -= 1.0 + z * z;
    f_aa1 *= xSq * xSq - 6.0 * xSq + 1.0;
    double f_aa2 = 0.5 + temp;
    f_aa2 *= z;
    f_aa2 *= 8 * x * (xSq - 1.0);
    double f_aa3 = (1.0 - 3 * xSq) * temp;
    f_aa3 -= x * y * z;
    f_aa3 += f_aa3;
    double f_aa = f_aa1 + f_aa2 + f_aa3;
    f_aa /= std::pow(y, 3);
    /// Hashed values of special functions for x = 2, 3, 4
    /// Gamma(x)
    static constexpr int gammaTable[] = {1, 2, 6};
    /// Gamma'(x)
    static constexpr long double gammaDerTable[] = {1.0 - M_EULER, 3.0 - 2.0 * M_EULER, 11.0 - 6.0 * M_EULER};
    /// Gamma''(x)
    static constexpr long double gammaSecDerTable[] = {0.82368066085287938958l, 2.49292999190269305794l, 11.1699273161019477314l};
    /// Digamma(x)
    static constexpr long double digammaTable[] = {1.0 - M_EULER, 1.5 - M_EULER, 11.0 / 6 - M_EULER};
    /// Digamma'(x)
    static constexpr long double digammaDerTable[] = {M_PI_SQ_6 - 1.0, M_PI_SQ_6 - 1.25, M_PI_SQ_6 - 49.0 / 36};
    /// Digamma''(x)
    static constexpr long double digammaSecDerTable[] = {-0.40411380631918857080l, -0.15411380631918857080l, -0.08003973224511449673l};
    /// third derivative
    double gTable[] = {0, 0, 0};
    for(int i = 0; i < 3; ++i)
    {
      double g_11 = 0.25 * gammaTable[i] * logY * logY;
      g_11 -= gammaDerTable[i] * logY;
      g_11 += gammaSecDerTable[i];
      double aux = digammaTable[i] - 0.5 * logY;
      double g_12 = aux;
      double zip2 = z * (i + 2);
      double cosZNu = std::cos(zip2), zSinZNu = z * std::sin(zip2);
      g_12 *= cosZNu;
      g_12 -= zSinZNu;
      double g_1 = g_11 * g_12;
      double g_21 = -gammaTable[i] * logY + 2 * gammaDerTable[i];
      double g_22 = -zSinZNu * aux;
      g_22 -= zSq * cosZNu;
      g_22 += cosZNu * digammaDerTable[i];
      double g_2 = g_21 * g_22;
      double g_3 = -zSq * cosZNu * aux;
      g_3 -= 2 * zSinZNu * digammaDerTable[i];
      g_3 += zSq * zSinZNu;
      g_3 += cosZNu * digammaSecDerTable[i];
      g_3 *= gammaTable[i];
      double g = g_1 + g_2 + g_3;
      g *= std::pow(y, -0.5 * i - 1);
      gTable[i] = g;
    }
    double f_aaa = -gTable[0] + 3 * gTable[1] - gTable[2];
    /// sum all three derivatives
    double tail = f_a * alpham1;
    tail += 0.5 * f_aa * alpham1 * alpham1;
    tail += std::pow(alpham1, 3) * f_aaa / 6.0;
    tail /= M_PI;
    return tail;
  }

  /**
   * @fn limitCaseForIntegrandAuxForGeneralExponent
   * @param theta
   * @param xiAdj
   * @return large values in the case of closeness to extreme points
   */
  double limitCaseForIntegrandAuxForGeneralExponent(double theta, double xiAdj) const
  {
    /// We got numerical error, need to investigate to which extreme point we are
    /// closer
    if(theta < 0.5 * (M_PI_2 - xiAdj))
      return alpha < 1 ? -BIG_NUMBER : BIG_NUMBER;
    return alpha < 1 ? BIG_NUMBER : -BIG_NUMBER;
  }

  /**
   * @fn integrandAuxForGeneralExponent
   * @param theta
   * @param xAdj
   * @param xiAdj
   * @return supplementary value for the integrand used in pdf calculations for
   * α ≠ 1
   */
  double integrandAuxForGeneralExponent(double theta, double xAdj, double xiAdj) const
  {
    if(std::fabs(theta) >= M_PI_2 || theta <= -xiAdj)
      return limitCaseForIntegrandAuxForGeneralExponent(theta, xiAdj);
    double thetaAdj = alpha * (theta + xiAdj);
    double sinThetaAdj = std::sin(thetaAdj);
    double y = std::log(std::cos(theta));
    y -= alpha * std::log(sinThetaAdj);
    y /= alpha - 1.0;
    y += std::log(std::cos(thetaAdj - theta));
    y += xAdj;
    return std::isfinite(y) ? y : limitCaseForIntegrandAuxForGeneralExponent(theta, xiAdj);
  }

  /**
   * @fn integrandFoGeneralExponent
   * @param theta
   * @param xAdj
   * @param xiAdj
   * @return the value of the integrand used for calculations of pdf for α ≠ 1
   */
  double integrandFoGeneralExponent(double theta, double xAdj, double xiAdj) const
  {
    if(std::fabs(theta) >= M_PI_2)
      return 0.0;
    if(theta <= -xiAdj)
      return 0.0;
    double u = integrandAuxForGeneralExponent(theta, xAdj, xiAdj);
    return fastpdfExponentiation(u);
  }

  /**
   * @fn pdfForGeneralExponent
   * @param x
   * @return value of probability density function for α ≠ 1
   */
  double pdfForGeneralExponent(double x) const
  {
    /// Standardize
    double xSt = (x - mu) / gamma;
    double absXSt = xSt;
    /// +- xi
    double xiAdj = xi;
    if(xSt > 0)
    {
      if(alpha < 1 && beta == -1)
        return 0.0;
    }
    else
    {
      if(alpha < 1 && beta == 1)
        return 0.0;
      absXSt = -xSt;
      xiAdj = -xi;
    }

    /// If α is too close to 1 and distribution is symmetric, then we approximate
    /// using Taylor series
    if(beta == 0.0 && std::fabs(alpha - 1.0) < 0.01)
      return pdfCauchy(x) + pdfTaylorExpansionTailNearCauchy(absXSt) / gamma;

    /// If x = 0, we know the analytic solution
    if(xSt == 0.0)
      return pdfAtZero() / gamma;

    double logAbsX = std::log(absXSt) - omega;

    /// If x is too close to 0, we do series expansion avoiding numerical problems
    if(logAbsX < seriesZeroParams.second)
    {
      if(alpha < 1 && std::fabs(beta) == 1)
        return pdfShortTailExpansionForGeneralExponent(logAbsX);
      return pdfSeriesExpansionAtZero(logAbsX, xiAdj, seriesZeroParams.first) / gamma;
    }

    /// If x is large enough we use tail approximation
    if(logAbsX > pdftailBound && alpha <= ALMOST_TWO)
    {
      if(alpha > 1 && std::fabs(beta) == 1)
        return pdfShortTailExpansionForGeneralExponent(logAbsX);
      return pdfSeriesExpansionAtInf(logAbsX, xiAdj) / gamma;
    }

    double xAdj = alpha_alpham1 * logAbsX;

    /// Search for the peak of the integrand
    double theta0 = 0;
    std::function<double(double)> funPtr = std::bind(&StableDistribution<RealType>::integrandAuxForGeneralExponent, this, std::placeholders::_1, xAdj, xiAdj);
    RandMath::findRootBrentFirstOrder<double>(funPtr, -xiAdj, M_PI_2, theta0);

    /// If theta0 is too close to π/2 or -xiAdj then we can still underestimate
    /// the integral
    int maxRecursionDepth = 11;
    double closeness = std::min(M_PI_2 - static_cast<long double>(theta0), static_cast<long double>(theta0) + static_cast<long double>(xiAdj));
    if(closeness < 0.1)
      maxRecursionDepth = 20;
    else if(closeness < 0.2)
      maxRecursionDepth = 15;

    /// Calculate sum of two integrals
    std::function<double(double)> integrandPtr = std::bind(&StableDistribution<RealType>::integrandFoGeneralExponent, this, std::placeholders::_1, xAdj, xiAdj);
    double int1 = RandMath::integral(integrandPtr, -xiAdj, theta0, 1e-11, maxRecursionDepth);
    double int2 = RandMath::integral(integrandPtr, theta0, M_PI_2, 1e-11, maxRecursionDepth);
    double res = pdfCoef * (int1 + int2) / absXSt;

    /// Finally we check if α is not too close to 2
    if(alpha <= ALMOST_TWO)
      return res;

    /// If α is near 2, we use tail aprroximation for large x
    /// and compare it with integral representation
    double alphap1 = alpha + 1.0;
    double tail = std::lgammal(alphap1);
    tail -= alphap1 * logAbsX;
    tail = std::exp(tail);
    tail *= (1.0 - 0.5 * alpha) / gamma;
    return std::max(tail, res);
  }

  /**
   * @fn fastcdfExponentiation
   * @param u
   * @return exp(-exp(u)), accelerated by truncation of input u
   */
  static double fastcdfExponentiation(double u)
  {
    if(u > 5.0)
      return 0.0;
    else if(u < -50.0)
      return 1.0;
    double y = std::exp(u);
    return std::exp(-y);
  }

  /**
   * @fn cdfAtZero
   * @param xiAdj adjusted ξ
   * @return cumulative distribution function for x = 0
   */
  double cdfAtZero(double xiAdj) const
  {
    return 0.5 - M_1_PI * xiAdj;
  }

  /**
   * @fn cdfForUnityExponent
   * @param x
   * @return cumulative distribution function for α = 1, β ≠ 0
   */
  double cdfForUnityExponent(double x) const
  {
    double xSt = (x - mu) / gamma;
    double xAdj = -M_PI_2 * xSt / beta - logGammaPi_2;
    double y = M_1_PI * RandMath::integral(
                            [this, xAdj](double theta) {
                              double u = integrandAuxForUnityExponent(theta, xAdj);
                              return fastcdfExponentiation(u);
                            },
                            -M_PI_2, M_PI_2);
    return (beta > 0) ? y : 1.0 - y;
  }

  /**
   * @fn cdfSeriesExpansionAtZero
   * @param logX log(x)
   * @param xiAdj adjusted ξ
   * @param k number of elements in series
   * @return series expansion of cumulative distribution function for x near 0
   */
  double cdfSeriesExpansionAtZero(double logX, double xiAdj, int k) const
  {
    /// Calculate first term of the sum
    /// (if x = 0, only this term is non-zero)
    double y0 = cdfAtZero(xiAdj);
    double sum = 0.0;
    if(beta == 0.0)
    {
      /// Symmetric distribution
      for(int m = 0; m <= k; ++m)
      {
        int m2p1 = 2 * m + 1;
        double term = std::lgammal(m2p1 * alphaInv);
        term += m2p1 * logX;
        term -= RandMath::lfact(m2p1);
        term = std::exp(term);
        sum += (m & 1) ? -term : term;
      }
    }
    else
    {
      /// Asymmetric distribution
      double rhoPi_alpha = M_PI_2 + xiAdj;
      for(int n = 1; n <= k; ++n)
      {
        double term = std::lgammal(n * alphaInv);
        term += n * logX;
        term -= RandMath::lfact(n);
        term = std::exp(term);
        term *= std::sin(n * rhoPi_alpha);
        sum += (n & 1) ? term : -term;
      }
    }
    return y0 + sum * M_1_PI * alphaInv;
  }

  /**
   * @fn pdfSeriesExpansionAtInf
   * @param logX log(x)
   * @param xiAdj adjusted ξ
   * @return series expansion of cumulative distribution function for large x
   */
  double cdfSeriesExpansionAtInf(double logX, double xiAdj) const
  {
    static constexpr int k = 10; /// number of elements in the series
    double rhoPi = M_PI_2 + xiAdj;
    rhoPi *= alpha;
    double sum = 0.0;
    for(int n = 1; n <= k; ++n)
    {
      double aux = n * alpha;
      double term = std::lgammal(aux);
      term -= aux * logX;
      term -= RandMath::lfact(n);
      term = std::exp(term);
      term *= std::sin(rhoPi * n);
      sum += (n & 1) ? term : -term;
    }
    return M_1_PI * sum;
  }

  /**
   * @fn cdfIntegralRepresentation
   * @param absXSt absolute value of standardised x
   * @param xiAdj adjusted ξ
   * @return integral representation of cumulative distribution function for
   * general case of α ≠ 1
   */
  double cdfIntegralRepresentation(double logX, double xiAdj) const
  {
    double xAdj = alpha_alpham1 * logX;
    return M_1_PI * RandMath::integral(
                        [this, xAdj, xiAdj](double theta) {
                          double u = integrandAuxForGeneralExponent(theta, xAdj, xiAdj);
                          return fastcdfExponentiation(u);
                        },
                        -xiAdj, M_PI_2);
  }

  /**
   * @fn cdfForGeneralExponent
   * @param x
   * @return cumulative distribution function for general case of α ≠ 1
   */
  double cdfForGeneralExponent(double x) const
  {
    double xSt = (x - mu) / gamma; /// Standardize
    if(xSt == 0)
      return cdfAtZero(xi);
    if(xSt > 0.0)
    {
      double logAbsX = std::log(xSt) - omega;
      /// If x is too close to 0, we do series expansion avoiding numerical
      /// problems
      if(logAbsX < seriesZeroParams.second)
        return cdfSeriesExpansionAtZero(logAbsX, xi, seriesZeroParams.first);
      /// If x is large enough we use tail approximation
      if(logAbsX > cdftailBound)
        return 1.0 - cdfSeriesExpansionAtInf(logAbsX, xi);
      if(alpha > 1.0)
        return 1.0 - cdfIntegralRepresentation(logAbsX, xi);
      return (beta == -1.0) ? 1.0 : cdfAtZero(xi) + cdfIntegralRepresentation(logAbsX, xi);
    }
    /// For x < 0 we use relation F(-x, xi) + F(x, -xi) = 1
    double logAbsX = std::log(-xSt) - omega;
    if(logAbsX < seriesZeroParams.second)
      return 1.0 - cdfSeriesExpansionAtZero(logAbsX, -xi, seriesZeroParams.first);
    if(logAbsX > cdftailBound)
      return cdfSeriesExpansionAtInf(logAbsX, -xi);
    if(alpha > 1.0)
      return cdfIntegralRepresentation(logAbsX, -xi);
    return (beta == 1.0) ? 0.0 : cdfAtZero(xi) - cdfIntegralRepresentation(logAbsX, -xi);
  }

  /**
   * @fn variateForUnityExponent
   * @return variate, generated by algorithm for α = 1, β ≠ 0
   */
  double variateForUnityExponent() const
  {
    RealType U = M_PI * UniformRand<RealType>::StandardVariate(this->localRandGenerator) - M_PI_2;
    RealType W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType pi_2pBetaU = M_PI_2 + beta * U;
    RealType Y = W * std::cos(U) / pi_2pBetaU;
    RealType X = std::log(Y);
    X += logGammaPi_2;
    X *= -beta;
    X += pi_2pBetaU * std::tan(U);
    X *= M_2_PI;
    return mu + gamma * X;
  }

  /**
   * @fn variateForGeneralExponent
   * @return variate, generated by algorithm for general case of α ≠ 1
   */
  double variateForGeneralExponent() const
  {
    RealType U = M_PI * randlib::UniformRand<RealType>::StandardVariate(this->localRandGenerator) - M_PI_2;
    RealType W = randlib::ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType alphaUpxi = alpha * (U + xi);
    RealType X = std::sin(alphaUpxi);
    RealType W_adj = W / std::cos(U - alphaUpxi);
    X *= W_adj;
    RealType R = omega - alphaInv * std::log(W_adj * std::cos(U));
    X *= std::exp(R);
    return mu + gamma * X;
  }

  /**
   * @fn variateForExponentEqualOneHalf
   * @return variate, generated by algorithm for special case of α = 0.5
   */
  double variateForExponentEqualOneHalf() const
  {
    RealType Z1 = randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType Z2 = randlib::NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    RealType temp1 = (1.0 + beta) / Z1, temp2 = (1.0 - beta) / Z2;
    RealType var = temp1 - temp2;
    var *= temp1 + temp2;
    var *= 0.25;
    return mu + gamma * var;
  }

  RealType quantileImpl(double p) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return quantileNormal(p);
    case CAUCHY:
      return quantileCauchy(p);
    case LEVY:
      return (beta > 0) ? quantileLevy(p) : 2 * mu - quantileLevy1m(p);
    default:
      return randlib::ContinuousDistribution<RealType>::quantileImpl(p);
    }
  }

  RealType quantileImpl1m(double p) const override
  {
    switch(distributionType)
    {
    case NORMAL:
      return quantileNormal1m(p);
    case CAUCHY:
      return quantileCauchy1m(p);
    case LEVY:
      return (beta > 0) ? quantileLevy1m(p) : 2 * mu - quantileLevy(p);
    default:
      return randlib::ContinuousDistribution<RealType>::quantileImpl1m(p);
    }
  }

  std::complex<double> CFImpl(double t) const override
  {
    double x = 0;
    switch(distributionType)
    {
    case NORMAL:
      return cfNormal(t);
    case CAUCHY:
      return cfCauchy(t);
    case LEVY: {
      std::complex<double> phi = cfLevy(t);
      return (beta > 0) ? phi : std::conj(phi);
    }
    case UNITY_EXPONENT:
      x = beta * M_2_PI * std::log(t);
      break;
    default:
      x = -zeta;
    }
    double re = std::pow(gamma * t, alpha);
    std::complex<double> psi = std::complex<double>(re, re * x - mu * t);
    return std::exp(-psi);
  }
};

/**
 * @brief The StableRand class <BR>
 * Stable distribution
 */
template <typename RealType = double>
class RANDLIB_EXPORT StableRand : public StableDistribution<RealType>
{
public:
  StableRand(double exponent = 2, double skewness = 0, double scale = 1, double location = 0)
  : StableDistribution<RealType>(exponent, skewness, scale, location)
  {
  }

  String Name() const override
  {
    return "Stable(" + this->toStringWithPrecision(this->GetExponent()) + ", " + this->toStringWithPrecision(this->GetSkewness()) + ", " + this->toStringWithPrecision(this->GetScale()) + ", " +
           this->toStringWithPrecision(this->GetLocation()) + ")";
  }

  void SetExponent(double exponent)
  {
    this->SetParameters(exponent, this->GetSkewness(), this->GetScale(), this->GetLocation());
  }

  void SetSkewness(double skewness)
  {
    this->SetParameters(this->GetExponent(), skewness, this->GetScale(), this->GetLocation());
  }
};

/**
 * @brief The HoltsmarkRand class <BR>
 * Holtsmark distribution
 *
 * Notation: X ~ Holtsmark(γ, μ)
 *
 * Related distributions:
 * X ~ S(1.5, 0, γ, μ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT HoltsmarkRand : public StableDistribution<RealType>
{
public:
  HoltsmarkRand(double scale = 1, double location = 0)
  : StableDistribution<RealType>(1.5, 0.0, scale, location)
  {
  }

  String Name() const override
  {
    return "Holtsmark(" + this->toStringWithPrecision(this->GetScale()) + ", " + this->toStringWithPrecision(this->GetLocation()) + ")";
  }
};

/**
 * @brief The LandauRand class <BR>
 * Landau distribution
 *
 * Notation: X ~ Landau(γ, μ)
 *
 * Related distributions:
 * X ~ S(1, 1, γ, μ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT LandauRand : public StableDistribution<RealType>
{
public:
  LandauRand(double scale = 1, double location = 0)
  : StableDistribution<RealType>(1.0, 1.0, scale, location)
  {
  }

  String Name() const override
  {
    return "Landau(" + this->toStringWithPrecision(this->GetScale()) + ", " + this->toStringWithPrecision(this->GetLocation()) + ")";
  }
};
} // namespace randlib
