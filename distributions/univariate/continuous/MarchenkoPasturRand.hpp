#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/BetaRand.hpp"

namespace randlib
{
/**
 * @brief The MarchenkoPasturRand class <BR>
 * Marchenko-Pastur distribution
 *
 * Notation: X ~ Marchenko-Pastur(λ, σ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT MarchenkoPasturRand : public randlib::ContinuousDistribution<RealType>
{
  double lambda = 1;    ///< ratio index λ
  double sigmaSq = 1;   ///< scale parameter σ^2
  double a = 0;         ///< minimum value (apart from 0 if λ > 1)
  double b = 4;         ///< maximum value
  double logLambda = 0; /// < log(λ)

  BetaRand<RealType> BetaRV{0.5, 1.5, 0, 4}; ///< beta-distributed rv for generator
  double M = 1.0;                            ///< rejection constant

public:
  MarchenkoPasturRand(double ratio = 1, double scale = 1)
  {
    SetParameters(ratio, scale);
  }

  String Name() const override
  {
    return "Marchenko-Pastur(" + this->toStringWithPrecision(GetRatio()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return (lambda < 1) ? sigmaSq * a : 0;
  }

  RealType MaxValue() const override
  {
    return sigmaSq * b;
  }

  void SetParameters(double ratio, double scale)
  {
    if(ratio <= 0.0)
      throw std::invalid_argument("Marchenko-Pastur distribution: ratio parameter should be positive");
    if(scale <= 0.0)
      throw std::invalid_argument("Marchenko-Pastur distribution: scale should be positive");
    lambda = ratio;
    double sqrtLambda = std::sqrt(lambda);
    a = 1.0 - sqrtLambda;
    a *= a;
    b = 1.0 + sqrtLambda;
    b *= b;

    sigmaSq = scale;
    logLambda = std::log(lambda);

    GENERATOR_ID genId = getIdOfUsedGenerator();
    if(genId == TINY_RATIO || genId == HUGE_RATIO)
    {
      BetaRV.SetShapes(1.5, 1.5);
      BetaRV.SetSupport(a, b);
      M = std::min(a / lambda, a);
    }
    else
    {
      BetaRV.SetShapes(0.5, 1.5);
      BetaRV.SetSupport(0, b);
      M = 0.25 * b / sqrtLambda;
      if(lambda > 1)
        M /= lambda * lambda;
    }
  }

  double GetRatio() const
  {
    return lambda;
  }

  double GetScale() const
  {
    return sigmaSq;
  }

  double f(const RealType& x) const override
  {
    if(x == 0.0)
      return (lambda < 1) ? 0.0 : INFINITY;
    double xSt = x / sigmaSq;
    if(xSt < a || xSt > b)
      return 0.0;
    double y = 0.5 * std::sqrt((b - xSt) * (xSt - a));
    y /= (lambda * xSt);
    y /= (M_PI * sigmaSq);
    return y;
  }

  double logf(const RealType& x) const override
  {
    if(x == 0.0)
      return (lambda < 1) ? -INFINITY : INFINITY;
    double xSt = x / sigmaSq;
    if(xSt < a || xSt > b)
      return -INFINITY;
    double y = 0.5 * std::log((b - xSt) * (xSt - a));
    y -= M_LN2 + M_LNPI + logLambda + std::log(x);
    return y;
  }

  double F(const RealType& x) const override
  {
    double xSt = x / sigmaSq;
    if(xSt < 0.0)
      return 0.0;
    if(xSt >= b)
      return 1.0;
    if(lambda > 1.0)
      return (xSt > a) ? 1.0 - ccdfForLargeRatio(xSt) : 1.0 - 1.0 / lambda;
    return (xSt > a) ? cdfForSmallRatio(xSt) : 0.0;
  }

  double S(const RealType& x) const override
  {
    double xSt = x / sigmaSq;
    if(xSt < 0.0)
      return 1.0;
    if(xSt >= b)
      return 0.0;
    if(lambda > 1.0)
      return (xSt > a) ? ccdfForLargeRatio(xSt) : 1.0 / lambda;
    return (xSt > a) ? 1.0 - cdfForSmallRatio(xSt) : 1.0;
  }

  RealType Variate() const override
  {
    switch(getIdOfUsedGenerator())
    {
    case TINY_RATIO:
      return sigmaSq * variateForTinyRatio();
    case SMALL_RATIO:
      return sigmaSq * variateForSmallRatio();
    case LARGE_RATIO:
      return sigmaSq * variateForLargeRatio();
    case HUGE_RATIO:
      return sigmaSq * variateForHugeRatio();
    default:
      throw std::runtime_error("Marchenko-Pastur distribution: invalid generator id");
    }
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    switch(getIdOfUsedGenerator())
    {
    case TINY_RATIO:
      for(RealType& var : outputData)
        var = sigmaSq * variateForTinyRatio();
      break;
    case SMALL_RATIO:
      for(RealType& var : outputData)
        var = sigmaSq * variateForSmallRatio();
      break;
    case LARGE_RATIO:
      for(RealType& var : outputData)
        var = sigmaSq * variateForLargeRatio();
      break;
    case HUGE_RATIO:
      for(RealType& var : outputData)
        var = sigmaSq * variateForHugeRatio();
      break;
    default:
      return;
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    BetaRV.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return sigmaSq;
  }

  long double Variance() const override
  {
    return sigmaSq * sigmaSq * lambda;
  }

  RealType Mode() const override
  {
    if(lambda > 1)
      return 0.0;
    RealType mode = lambda - 1.0;
    mode *= mode;
    mode /= lambda + 1.0;
    return sigmaSq * mode;
  }

  long double Skewness() const override
  {
    long double mu = Mean();
    long double var = Variance();
    long double skewness = Moment(3);
    skewness -= std::pow(mu, 3);
    skewness -= 3 * mu * var;
    skewness /= std::pow(var, 1.5);
    return skewness;
  }

  long double ExcessKurtosis() const override
  {
    long double mu = Mean();
    long double var = Variance();
    long double moment3 = Moment(3);
    long double kurtosis = Moment(4);
    long double muSq = mu * mu;
    kurtosis -= 4 * mu * moment3;
    kurtosis += 3 * muSq * muSq;
    kurtosis += 6 * muSq * var;
    kurtosis /= var * var;
    return kurtosis - 3.0;
  }

private:
  /**
   * @fn ccdfForLargeRatio
   * @param x
   * @return S(x) for λ > 1
   */
  double ccdfForLargeRatio(const RealType& x) const
  {
    double y1 = 1.0 - x + lambda;
    double lambdam1 = lambda - 1.0;
    double lambdap1 = lambda + 1.0;
    double temp = std::sqrt(4 * lambda - y1 * y1);
    if(temp != 0.0)
    {
      y1 /= temp;
      y1 = lambdap1 * RandMath::atan(y1);
    }
    else
      y1 = RandMath::sign(y1) * M_PI_2;
    double y2 = x * lambdap1;
    y2 -= lambdam1 * lambdam1;
    y2 /= temp * lambdam1;
    y2 = lambdam1 * RandMath::atan(y2);
    double y = M_PI - temp + y1 + y2;
    y /= M_PI * lambda;
    return 0.5 * y;
  }

  /**
   * @fn cdfForSmallRatio
   * @param x
   * @return F(x) for 0 < λ <= 1
   */
  double cdfForSmallRatio(const RealType& x) const
  {
    double y1 = 1.0 - x + lambda;
    double temp = std::sqrt(4 * lambda - y1 * y1);
    double lambdam1 = lambda - 1.0;
    double lambdap1 = lambda + 1.0;
    if(temp != 0.0)
    {
      y1 /= temp;
      y1 = lambdap1 * RandMath::atan(y1);
    }
    else
      y1 = RandMath::sign(y1) * M_PI_2;
    double y2 = 0.0;
    if(lambdam1 != 0)
    {
      y2 = x * lambdap1;
      y2 -= lambdam1 * lambdam1;
      y2 /= -temp * lambdam1;
      y2 = lambdam1 * RandMath::atan(y2);
    }
    double y = M_PI * lambda + temp - y1 + y2;
    y /= M_PI * lambda;
    return 0.5 * y;
  }

  enum GENERATOR_ID
  {
    TINY_RATIO,
    SMALL_RATIO,
    LARGE_RATIO,
    HUGE_RATIO
  };

  GENERATOR_ID getIdOfUsedGenerator() const
  {
    if(lambda < 0.3)
      return TINY_RATIO;
    if(lambda <= 1.0)
      return SMALL_RATIO;
    return (lambda > 3.3) ? HUGE_RATIO : LARGE_RATIO;
  }

  RealType variateForTinyRatio() const
  {
    size_t iter = 0;
    do
    {
      double X = BetaRV.Variate();
      double U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      if(U < M / X)
        return X;
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Marchenko-Pastur distribution: sampling failed");
  }

  RealType variateForSmallRatio() const
  {
    size_t iter = 0;
    do
    {
      double X = BetaRV.Variate();
      double U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      double ratio = M * (1.0 - a / X);
      if(U * U < ratio)
        return X;
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Marchenko-Pastur distribution: sampling failed");
  }

  RealType variateForLargeRatio() const
  {
    return (UniformRand<RealType>::StandardVariate(this->localRandGenerator) > 1.0 / lambda) ? 0.0 : variateForSmallRatio();
  }

  RealType variateForHugeRatio() const
  {
    return (UniformRand<RealType>::StandardVariate(this->localRandGenerator) > 1.0 / lambda) ? 0.0 : variateForTinyRatio();
  }

  long double Moment(int n) const
  {
    if(n < 0)
      throw std::invalid_argument("Marchenko-Pastur distribution: degree n should be non-negative");
    switch(n)
    {
    case 0:
      return 1.0;
    case 1:
      return sigmaSq;
    case 2:
      return sigmaSq * sigmaSq * lambda;
    default: {
      long double sum = 0.0;
      long double temp = RandMath::lfact(n) + RandMath::lfact(n - 1);
      long double nlogSigmaSq = n * std::log(sigmaSq);
      for(int k = 0; k != n; ++k)
      {
        long double term = temp;
        int kp1 = k + 1;
        term -= 2 * RandMath::lfact(k);
        term -= RandMath::lfact(n - k);
        term -= RandMath::lfact(n - kp1);
        term += k * logLambda;
        term += nlogSigmaSq;
        term = std::exp(term) / kp1;
        sum += term;
      }
      return sum;
    }
    }
  }

  RealType quantileImpl(double p) const override
  {
    if(p <= 1.0 - 1.0 / lambda)
      return 0.0;
    RealType minInitValue = sigmaSq * a;
    RealType maxInitValue = sigmaSq * b;
    RealType initValue = minInitValue + p * (maxInitValue - minInitValue);
    return randlib::ContinuousDistribution<RealType>::quantileImpl(p, initValue);
  }

  RealType quantileImpl1m(double p) const override
  {
    if(p >= 1.0 / lambda)
      return 0.0;
    RealType minInitValue = sigmaSq * a;
    RealType maxInitValue = sigmaSq * b;
    RealType initValue = maxInitValue - p * (maxInitValue - minInitValue);
    return randlib::ContinuousDistribution<RealType>::quantileImpl1m(p, initValue);
  }

  std::complex<double> CFImpl(double t) const override
  {
    if(lambda < 1)
      return randlib::ContinuousDistribution<RealType>::CFImpl(t);
    /// otherwise we have singularity at point 0
    if(lambda == 1)
    {
      /// we split integrand for real part on (cos(tx)-1)f(x) and f(x)
      double re = this->ExpectedValue([this, t](double x) { return std::cos(t * x) - 1.0; }, 0, 4 * sigmaSq);
      double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, 0, 4 * sigmaSq);
      return std::complex<double>(1.0 + re, im);
    }
    /// for λ > 1 we split integral on 2 parts: at point 0 and the rest
    double re = this->ExpectedValue([this, t](double x) { return std::cos(t * x); }, sigmaSq * a, sigmaSq * b);
    double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, sigmaSq * a, sigmaSq * b);
    return std::complex<double>(1.0 - 1.0 / lambda + re, im);
  }
};
} // namespace randlib
