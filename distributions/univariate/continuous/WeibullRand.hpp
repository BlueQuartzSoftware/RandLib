#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/InverseGammaRand.hpp"

namespace randlib
{
/**
 * @brief The WeibullRand class <BR>
 * Weibull distribution
 *
 * Notation: X ~ Weibull(λ, k)
 */
template <typename RealType = double>
class RANDLIB_EXPORT WeibullRand : public randlib::ContinuousDistribution<RealType>
{
  double lambda = 1;      ///< scale λ
  double k = 1;           ///< shape k
  double kInv = 1;        /// 1 /k
  double logk_lambda = 0; /// log(k/λ)

public:
  WeibullRand(double scale = 1, double shape = 1)
  {
    SetParameters(scale, shape);
  }

  String Name() const override
  {
    return "Weibull(" + this->toStringWithPrecision(GetScale()) + ", " + this->toStringWithPrecision(GetShape()) + ")";
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

  void SetParameters(double scale, double shape)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Weibull distribution: scale should be positive");
    if(shape <= 0.0)
      throw std::invalid_argument("Weibull distribution: shape should be positive");
    lambda = scale;
    k = shape;
    kInv = 1.0 / k;
    logk_lambda = std::log(k / lambda);
  }

  inline double GetScale() const
  {
    return lambda;
  }

  inline double GetShape() const
  {
    return k;
  }

  double f(const RealType& x) const override
  {
    if(x < 0)
      return 0;
    if(x == 0)
    {
      if(k == 1)
        return 1.0 / lambda;
      return (k > 1) ? 0.0 : INFINITY;
    }
    return std::exp(this->logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < 0)
      return -INFINITY;
    if(x == 0)
    {
      if(k == 1)
        return logk_lambda;
      return (k > 1) ? -INFINITY : INFINITY;
    }
    double xAdj = x / lambda;
    double xAdjPow = std::pow(xAdj, k);
    return logk_lambda + (k - 1) * std::log(xAdj) - xAdjPow;
  }

  double F(const RealType& x) const override
  {
    return (x > 0.0) ? -std::expm1l(-std::pow(x / lambda, k)) : 0.0;
  }

  double S(const RealType& x) const override
  {
    return (x > 0.0) ? std::exp(-std::pow(x / lambda, k)) : 1.0;
  }

  RealType Variate() const override
  {
    return lambda * std::pow(ExponentialRand<RealType>::StandardVariate(this->localRandGenerator), kInv);
  }

  long double Mean() const override
  {
    return lambda * std::tgammal(1 + kInv);
  }

  long double Variance() const override
  {
    double res = std::tgammal(1 + kInv);
    res *= -res;
    res += std::tgammal(1 + kInv + kInv);
    return lambda * lambda * res;
  }

  RealType Median() const override
  {
    return lambda * std::pow(M_LN2, kInv);
  }

  RealType Mode() const override
  {
    if(k <= 1)
      return 0;
    double y = std::log1pl(-kInv);
    y = std::exp(kInv * y);
    return lambda * y;
  }

  long double Skewness() const override
  {
    long double mu = Mean();
    long double var = Variance();
    long double sigma = std::sqrt(var);
    long double numerator = std::tgammal(1 + 3.0 * kInv);
    numerator *= lambda * lambda * lambda;
    numerator -= 3 * mu * var;
    numerator -= mu * mu * mu;
    long double denominator = var * sigma;
    return numerator / denominator;
  }

  long double ExcessKurtosis() const override
  {
    long double mu = Mean();
    long double var = Variance();
    long double sigma = std::sqrt(var);
    long double skewness = Skewness();
    long double numerator = lambda * lambda;
    numerator *= numerator;
    numerator *= std::tgammal(1 + 4.0 * kInv);
    numerator -= 4 * skewness * var * sigma * mu;
    long double mu2 = mu * mu;
    numerator -= 6 * mu2 * var;
    numerator -= mu2 * mu2;
    long double kurtosis = numerator / (var * var);
    return kurtosis - 3;
  }

  long double Entropy() const
  {
    return M_EULER * (1.0 - kInv) + std::log(lambda * kInv) + 1.0;
  }

  /**
   * @fn FitScale
   * Fit λ by maximum-likelihood
   * @param sample
   */
  void FitScale(const std::vector<RealType>& sample)
  {
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    double knorm = getNorm(sample);
    SetParameters(knorm, k);
  }

  /**
   * @fn FitScaleBayes
   * Fit λ, using bayesian inference
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  InverseGammaRand<RealType> FitScaleBayes(const std::vector<RealType>& sample, const InverseGammaRand<RealType>& priorDistribution, bool MAP = false)
  {
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    int n = sample.size();
    double newShape = priorDistribution.GetShape() + n;
    double knorm = getNorm(sample);
    double newRate = priorDistribution.GetRate() + n * std::pow(knorm, k);
    InverseGammaRand<RealType> posteriorDistribution(newShape, newRate);
    double powScale = MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean();
    SetParameters(std::pow(powScale, kInv), k);
    return posteriorDistribution;
  }

private:
  RealType quantileImpl(double p) const override
  {
    double x = -std::log1pl(-p);
    x = std::pow(x, kInv);
    return lambda * x;
  }

  RealType quantileImpl1m(double p) const override
  {
    double x = -std::log(p);
    x = std::pow(x, kInv);
    return lambda * x;
  }

  std::complex<double> CFImpl(double t) const override
  {
    double lambdaT = lambda * t;
    if(k >= 1)
    {
      if(lambdaT > 0.5)
        return randlib::ContinuousDistribution<RealType>::CFImpl(t);
      /// for λt < 0.5, the worst case scenario for series expansion is n ~ 70
      long double re = 0.0, im = 0.0;
      long double addon = 0.0;
      long double logLambdaT = std::log(lambdaT);
      /// Series representation for real part
      int n = 0;
      do
      {
        int n2 = n + n;
        addon = n2 * logLambdaT;
        addon += std::lgammal(1.0 + n2 / k);
        addon -= std::lgammal(1.0 + n2);
        addon = std::exp(addon);
        re += (n & 1) ? -addon : addon;
        ++n;
      } while(std::fabs(addon) > MIN_POSITIVE * std::fabs(re));
      /// Series representation for imaginary part
      n = 0;
      do
      {
        int n2p1 = n + n + 1;
        addon = n2p1 * logLambdaT;
        addon += std::lgammal(1.0 + n2p1 / k);
        addon -= std::lgammal(1.0 + n2p1);
        addon = std::exp(addon);
        im += (n & 1) ? -addon : addon;
        ++n;
      } while(std::fabs(addon) > MIN_POSITIVE * std::fabs(im));
      return std::complex<double>(re, im);
    }

    /// For real part with k < 1 we split the integral on two intervals
    double re1 = RandMath::integral(
        [this, t](double x) {
          if(x <= 0.0 || x > 1.0)
            return 0.0;
          double xAdj = x / lambda;
          double xAdjPow = std::pow(xAdj, k - 1);
          double y = k / lambda * xAdjPow * std::expm1l(-xAdj * xAdjPow);
          return std::cos(t * x) * y;
        },
        0.0, 1.0);

    double re2 = this->ExpectedValue([this, t](double x) { return std::cos(t * x); }, 1.0, INFINITY);

    double re3 = t * RandMath::integral(
                         [this, t](double x) {
                           if(x <= 0.0)
                             return 0.0;
                           return std::sin(t * x) * std::pow(x, k);
                         },
                         0.0, 1.0);
    re3 += std::cos(t);
    re3 /= std::pow(lambda, k);

    double re = re1 + re2 + re3;

    double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, 0.0, INFINITY);

    return std::complex<double>(re, im);
  }

  /**
   * @brief getNorm
   * @param sample
   * @return L^k norm of the sample
   */
  double getNorm(const std::vector<RealType>& sample) const
  {
    long double sum = 0;
    RealType maxVar = *std::max_element(sample.begin(), sample.end());
    for(const RealType& var : sample)
    {
      sum += std::pow(var / maxVar, k);
    }
    long double avg = sum / sample.size();
    return maxVar * std::pow(avg, kInv);
  }
};
} // namespace randlib
