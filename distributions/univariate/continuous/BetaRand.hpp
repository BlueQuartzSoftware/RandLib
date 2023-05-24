#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The BetaDistribution class <BR>
 * Abstract class for Beta distribution
 *
 * f(x | α, β) = y^{α-1} (1-y)^{β-1} / B(α, β), <BR>
 * where y = (x-a)/(b-a) and B(α, β) denotes Beta function
 *
 * Notation: X ~ B(α, β, a, b) or X ~ B(α, β) for a=0 and b=1
 *
 * Related distributions (a=0, b=1): <BR>
 * 1 − X ~ B(β, α) <BR>
 * X / (1 - X) ~ B'(α, β) <BR>
 * X = Y / (Y + Z), where Y ~ Γ(α) and Z ~ Γ(β) <BR>
 * βX / α(1 - X) ~ F(2α, 2β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT BetaDistribution : public randlib::ContinuousDistribution<RealType>
{
protected:
  double alpha = 1;  ///< first shape α
  double beta = 1;   ///< second shape β
  double a = 0;      ///< min bound
  double b = 1;      ///< max bound
  double bma = 1;    ///< b-a
  double bmaInv = 1; ///< 1/(b-a)
  double logbma = 0; ///< log(b-a)

  randlib::GammaRand<RealType> GammaRV1 = randlib::GammaRand<RealType>();
  randlib::GammaRand<RealType> GammaRV2 = randlib::GammaRand<RealType>();

private:
  static constexpr double edgeForGenerators = 8.0;
  double logBetaFun = 0; ///< log(B(α, β)
  double betaFun = 1;    ///< B(α, β)

  /// coefficients for generators
  struct genCoef_t
  {
    double s, t, u;
  } genCoef = {0, 0, 0};

protected:
  BetaDistribution(double shape1 = 1, double shape2 = 1, double minValue = 0, double maxValue = 1)
  {
    SetShapes(shape1, shape2);
    SetSupport(minValue, maxValue);
  }

  virtual ~BetaDistribution()
  {
  }

public:
  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return a;
  }

  RealType MaxValue() const override
  {
    return b;
  }

  /**
   * @fn GetAlpha
   * @return first shape α
   */
  inline double GetAlpha() const
  {
    return alpha;
  }

  /**
   * @fn GetBeta
   * @return second shape β
   */
  inline double GetBeta() const
  {
    return beta;
  }

  /**
   * @fn GetBetaFunction
   * @return B(α, β)
   */
  inline double GetBetaFunction() const
  {
    return betaFun;
  }

  /**
   * @fn GetLogBetaFunction
   * @return log(B(α, β))
   */
  inline double GetLogBetaFunction() const
  {
    return logBetaFun;
  }

  double f(const RealType& x) const override
  {
    if(x < a || x > b)
      return 0.0;
    if(x == a)
    {
      if(alpha == 1)
        return beta / bma;
      return (alpha > 1) ? 0 : INFINITY;
    }
    if(x == b)
    {
      if(beta == 1)
        return alpha / bma;
      return (beta > 1) ? 0 : INFINITY;
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    /// Standardize
    double xSt = (x - a) * bmaInv;
    if(xSt < 0.0 || xSt > 1.0)
      return -INFINITY;
    if(xSt == 0.0)
    {
      if(alpha == 1)
        return std::log(beta / bma);
      return (alpha > 1) ? -INFINITY : INFINITY;
    }
    if(xSt == 1.0)
    {
      if(beta == 1)
        return std::log(alpha / bma);
      return (beta > 1) ? -INFINITY : INFINITY;
    }
    double y = (alpha - 1) * std::log(xSt);
    y += (beta - 1) * std::log1pl(-xSt);
    return y - logBetaFun - logbma;
  }

  double F(const RealType& x) const override
  {
    if(x <= a)
      return 0.0;
    if(x >= b)
      return 1.0;
    /// Standardize
    double xSt = (x - a) * bmaInv;
    /// Workaround known case
    if(alpha == beta && beta == 0.5)
      return M_2_PI * std::asin(std::sqrt(xSt));
    return RandMath::ibeta(xSt, alpha, beta, logBetaFun, std::log(xSt), std::log1pl(-xSt));
  }

  double S(const RealType& x) const override
  {
    if(x <= a)
      return 1.0;
    if(x >= b)
      return 0.0;
    /// Standardize
    double xSt = (x - a) / bma;
    /// Workaround known case
    if(alpha == beta && beta == 0.5)
      return M_2_PI * std::acos(std::sqrt(xSt));
    return RandMath::ibeta(1.0 - xSt, beta, alpha, logBetaFun, std::log1pl(-xSt), std::log(xSt));
  }

  RealType Variate() const override
  {
    double var = 0;
    GENERATOR_ID id = getIdOfUsedGenerator();

    switch(id)
    {
    case UNIFORM:
      var = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      break;
    case ARCSINE:
      var = variateArcsine();
      break;
    case CHENG:
      var = variateCheng();
      break;
    case REJECTION_UNIFORM:
      var = variateRejectionUniform();
      break;
    case REJECTION_UNIFORM_EXTENDED:
      var = variateRejectionUniformExtended();
      break;
    case REJECTION_NORMAL:
      var = variateRejectionNormal();
      break;
    case JOHNK:
      var = variateJohnk();
      break;
    case ATKINSON_WHITTAKER:
      var = variateAtkinsonWhittaker();
      break;
    case GAMMA_RATIO:
    default:
      var = variateGammaRatio();
      break;
    }

    return a + bma * var;
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    GENERATOR_ID id = getIdOfUsedGenerator();

    switch(id)
    {
    case UNIFORM: {
      for(RealType& var : outputData)
        var = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    break;
    case ARCSINE: {
      for(RealType& var : outputData)
        var = variateArcsine();
    }
    break;
    case CHENG: {
      for(RealType& var : outputData)
        var = variateCheng();
    }
    break;
    case REJECTION_UNIFORM: {
      for(RealType& var : outputData)
        var = variateRejectionUniform();
    }
    break;
    case REJECTION_UNIFORM_EXTENDED: {
      for(RealType& var : outputData)
        var = variateRejectionUniformExtended();
    }
    break;
    case REJECTION_NORMAL: {
      for(RealType& var : outputData)
        var = variateRejectionNormal();
    }
    break;
    case JOHNK: {
      for(RealType& var : outputData)
        var = variateJohnk();
    }
    break;
    case ATKINSON_WHITTAKER: {
      for(RealType& var : outputData)
        var = variateAtkinsonWhittaker();
    }
    break;
    case GAMMA_RATIO:
    default: {
      GammaRV1.Sample(outputData);
      for(RealType& var : outputData)
        var /= (var + GammaRV2.Variate());
    }
    break;
    }

    /// Shift and scale
    for(RealType& var : outputData)
      var = a + bma * var;
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    GammaRV1.Reseed(seed + 1);
    GammaRV2.Reseed(seed + 2);
  }

  long double Mean() const override
  {
    double mean = alpha / (alpha + beta);
    return a + bma * mean;
  }

  /**
   * @fn GeometricMean
   * @return E[ln(X)]
   */
  long double GeometricMean() const
  {
    return RandMath::digamma(alpha) - RandMath::digamma(alpha + beta);
  }

  long double Variance() const override
  {
    double var = alpha + beta;
    var *= var * (var + 1);
    var = alpha * beta / var;
    return bma * bma * var;
  }

  /**
   * @fn GeometricVariance
   * @return Var(ln(X))
   */
  long double GeometricVariance() const
  {
    return RandMath::trigamma(alpha) - RandMath::trigamma(alpha + beta);
  }

  RealType Median() const override
  {
    if(alpha == beta)
      return a + bma * 0.5;
    if(alpha == 1.0)
      return a - bma * std::expm1l(-M_LN2 / beta);
    if(beta == 1.0)
      return a + bma * std::exp(-M_LN2 / alpha);
    if(alpha >= 1.0 && beta >= 1.0)
    {
      double initValue = 3 * alpha - 1.0;
      initValue /= 3 * (alpha + beta) - 2.0;
      initValue *= bma;
      initValue += a;
      return randlib::ContinuousDistribution<RealType>::quantileImpl(0.5, initValue);
    }
    return randlib::ContinuousDistribution<RealType>::quantileImpl(0.5);
  }

  RealType Mode() const override
  {
    double mode;
    if(alpha > 1)
      mode = (beta > 1) ? (alpha - 1) / (alpha + beta - 2) : 1.0;
    else
      mode = (beta > 1) ? 0.0 : (alpha > beta);
    return a + bma * mode;
  }

  long double Skewness() const override
  {
    long double skewness = (alpha + beta + 1) / (alpha * beta);
    skewness = std::sqrt(skewness);
    skewness *= beta - alpha;
    skewness /= alpha + beta + 2;
    return 2 * skewness;
  }

  long double ExcessKurtosis() const override
  {
    long double sum = alpha + beta;
    long double kurtosis = alpha - beta;
    kurtosis *= kurtosis;
    kurtosis *= (sum + 1);
    kurtosis /= (alpha * beta * (sum + 2));
    --kurtosis;
    kurtosis /= (sum + 3);
    return 6 * kurtosis;
  }

  /**
   * @brief MeanAbsoluteDeviation
   * @return E[|X - E[X]|]
   */
  long double MeanAbsoluteDeviation() const
  {
    double y = M_LN2;
    y += alpha * std::log(alpha);
    y += beta * std::log(beta);
    y -= (alpha + beta + 1) * std::log(alpha + beta);
    y -= logBetaFun;
    y += logbma;
    return std::exp(y);
  }

protected:
  /**
   * @fn SetShapes
   * @param shape1 α
   * @param shape2 β
   */
  void SetShapes(double shape1, double shape2)
  {
    if(shape1 <= 0 || shape2 <= 0)
      throw std::invalid_argument("Beta distribution: shapes should be positive");
    GammaRV1.SetParameters(shape1, 1);
    GammaRV2.SetParameters(shape2, 1);
    alpha = GammaRV1.GetShape();
    beta = GammaRV2.GetShape();
    betaFun = std::betal(alpha, beta);
    logBetaFun = std::log(betaFun);
    setCoefficientsForGenerator();
  }

  /**
   * @fn SetSupport
   * @param minValue a
   * @param maxValue b
   */
  void SetSupport(double minValue, double maxValue)
  {
    if(minValue >= maxValue)
      throw std::invalid_argument("Beta distribution: minimum value should be "
                                  "smaller than maximum value");

    a = minValue;
    b = maxValue;
    bma = b - a;
    bmaInv = 1.0 / bma;
    logbma = std::log(bma);
  }

  RealType quantileImpl(double p) const override
  {
    if(alpha == beta)
    {
      if(alpha == 0.5)
      {
        double x = std::sin(0.5 * M_PI * p);
        return a + bma * x * x;
      }
      if(alpha == 1.0)
        return a + bma * p;
    }
    if(alpha == 1.0)
      return a - bma * std::expm1l(std::log1pl(-p) / beta);
    if(beta == 1.0)
      return a + bma * std::pow(p, 1.0 / alpha);
    return randlib::ContinuousDistribution<RealType>::quantileImpl(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    if(alpha == beta)
    {
      if(alpha == 0.5)
      {
        double x = std::cos(0.5 * M_PI * p);
        return a + bma * x * x;
      }
      if(alpha == 1.0)
        return b - bma * p;
    }
    if(alpha == 1.0)
      return a - bma * std::expm1l(std::log(p) / beta);
    if(beta == 1.0)
      return a + bma * std::exp(std::log1pl(-p) / alpha);
    return randlib::ContinuousDistribution<RealType>::quantileImpl1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    /// if we don't have singularity points, we can use direct integration
    if(alpha >= 1 && beta >= 1)
      return randlib::UnivariateDistribution<RealType>::CFImpl(t);

    double z = bma * t;
    double sinZ = std::sin(z);
    double cosZm1 = std::cos(z) - 1.0;

    double re = RandMath::integral(
        [this, z, cosZm1](double x) {
          if(x >= 1)
            return 0.0;
          if(x <= 0)
            return -cosZm1;
          double y = std::cos(z * x) - 1;
          y *= std::pow(x, alpha - 1);
          y -= cosZm1;
          return std::pow(1.0 - x, beta - 1) * y;
        },
        0, 1);
    re += betaFun;
    re += cosZm1 / beta;

    double im = RandMath::integral(
        [this, z, sinZ](double x) {
          if(x >= 1)
            return 0.0;
          if(x <= 0)
            return -sinZ;
          double y = std::sin(z * x);
          y *= std::pow(x, alpha - 1);
          y -= sinZ;
          return std::pow(1.0 - x, beta - 1) * y;
        },
        0, 1);
    im += sinZ / beta;

    std::complex<double> y(re, im);
    double cosTA = std::cos(t * a), sinTA = std::sin(t * a);
    return y * std::complex<double>(cosTA, sinTA) / betaFun;
  }

  static constexpr char ALPHA_ZERO[] = "Possibly one or more elements of the sample coincide with the lower "
                                       "boundary a.";
  static constexpr char BETA_ZERO[] = "Possibly one or more elements of the sample coincide with the upper "
                                      "boundary b.";

private:
  enum GENERATOR_ID
  {
    UNIFORM,           ///< standard uniform variate
    ARCSINE,           ///< arcsine method
    CHENG,             ///< Cheng's method
    REJECTION_UNIFORM, ///< rejection method from uniform distribution for
    ///< specific value of shapes α = β = 1.5
    REJECTION_UNIFORM_EXTENDED, ///< rejection method from uniform distribution
    ///< accelerated by using exponential
    ///< distribution
    REJECTION_NORMAL,   ///< rejection method normal distribution
    JOHNK,              ///< Johnk's method
    ATKINSON_WHITTAKER, ///< Atkinson-Whittaker's method
    GAMMA_RATIO         ///< ratio of two gamma variables
  };

  /**
   * @fn getIdOfUsedGenerator
   * @return id of used variate generator according to the shapes
   */
  GENERATOR_ID getIdOfUsedGenerator() const
  {
    if(alpha < 1 && beta < 1 && alpha + beta > 1)
      return ATKINSON_WHITTAKER;

    if(RandMath::areClose(alpha, beta))
    {
      if(RandMath::areClose(alpha, 1.0))
        return UNIFORM;
      else if(RandMath::areClose(alpha, 0.5))
        return ARCSINE;
      else if(RandMath::areClose(alpha, 1.5))
        return REJECTION_UNIFORM;
      else if(alpha > 1)
        return (alpha < 2) ? REJECTION_UNIFORM_EXTENDED : REJECTION_NORMAL;
    }
    if(std::min(alpha, beta) > 0.5 && std::max(alpha, beta) > 1)
      return CHENG;
    return (alpha + beta < 2) ? JOHNK : GAMMA_RATIO;
  }

  /**
   * @fn setCoefficientsForGenerator
   */
  void setCoefficientsForGenerator()
  {
    GENERATOR_ID id = getIdOfUsedGenerator();
    if(id == REJECTION_NORMAL)
    {
      double alpham1 = alpha - 1;
      genCoef.s = alpham1 * std::log1pl(0.5 / alpham1) - 0.5;
      genCoef.t = 1.0 / std::sqrt(8 * alpha - 4);
    }
    else if(id == CHENG)
    {
      genCoef.s = alpha + beta;
      genCoef.t = std::min(alpha, beta);
      if(genCoef.t > 1)
        genCoef.t = std::sqrt((2 * alpha * beta - genCoef.s) / (genCoef.s - 2));
      genCoef.u = alpha + genCoef.t;
    }
    else if(id == ATKINSON_WHITTAKER)
    {
      genCoef.t = std::sqrt(alpha * (1 - alpha));
      genCoef.t /= (genCoef.t + std::sqrt(beta * (1 - beta)));
      genCoef.s = beta * genCoef.t;
      genCoef.s /= (genCoef.s + alpha * (1 - genCoef.t));
    }
  }

  /**
   * @fn variateRejectionUniform
   * Symmetric beta generator via rejection from the uniform density
   * @return beta variate for α = β = 1.5
   */
  RealType variateRejectionUniform() const
  {
    size_t iter = 0;
    do
    {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType V = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      if(0.25 * V * V <= U - U * U)
        return U;
    } while(++iter <= this->MAX_ITER_REJECTION);
    throw std::runtime_error("Beta distribution: sampling failed");
  }

  /**
   * @fn variateRejectionUniform
   * Symmetric beta generator via rejection from the uniform density
   * @return beta variate for 1 < α = β < 2 and α != 1.5
   */
  RealType variateRejectionUniformExtended() const
  {
    size_t iter = 0;
    static constexpr double M_LN4 = M_LN2 + M_LN2;
    do
    {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType edge = M_LN4 + std::log(U - U * U);
      if(W >= (1.0 - alpha) * edge)
        return U;
    } while(++iter <= this->MAX_ITER_REJECTION);
    throw std::runtime_error("Beta distribution: sampling failed");
  }

  /**
   * @fn variateArcsine
   * Arcsine beta generator
   * @return beta variate for α = β = 0.5
   */
  RealType variateArcsine() const
  {
    RealType U = 2 * UniformRand<RealType>::StandardVariate(this->localRandGenerator) - 1;
    RealType X = std::sin(M_PI * U);
    return X * X;
  }

  /**
   * @fn variateRejectionNormal
   * Symmetric beta generator via rejection from the normal density
   * @return beta variate for equal shape parameters > 2
   */
  RealType variateRejectionNormal() const
  {
    size_t iter = 0;
    RealType N = 0, Z = 0;
    RealType alpham1 = alpha - 1;
    RealType alpha2m1 = alpha + alpham1;
    do
    {
      do
      {
        N = NormalRand<RealType>::StandardVariate(this->localRandGenerator);
        Z = N * N;
      } while(Z >= alpha2m1);

      RealType W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator) + genCoef.s;
      RealType aux = 0.5 - alpham1 / (alpha2m1 - Z);
      aux *= Z;
      if(W + aux >= 0)
        return 0.5 + N * genCoef.t;
      aux = std::log1pl(-Z / alpha2m1);
      aux *= alpham1;
      aux += W + 0.5 * Z;
      if(aux >= 0)
        return 0.5 + N * genCoef.t;
    } while(++iter <= this->MAX_ITER_REJECTION);
    throw std::runtime_error("Beta distribution: sampling failed");
  }

  /**
   * @fn variateJohnk
   * Johnk's beta generator
   * @return beta variate for small shape parameters < 1
   */
  RealType variateJohnk() const
  {
    RealType X = 0, Z = 0;
    RealType W = 0, V = 0;
    do
    {
      W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator) / alpha;
      V = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator) / beta;
      X = std::exp(-W);
      Z = X + std::exp(-V);
    } while(Z > 1);
    return (Z > 0) ? (X / Z) : (W < V);
  }

  /**
   * @fn variateCheng
   * Cheng's beta generator
   * @return beta variate for max(α, β) > 1 and min(α, β) > 0.5
   */
  RealType variateCheng() const
  {
    RealType R, T, Y;
    do
    {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType V = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType X = std::log(U / (1 - U)) / genCoef.t;
      Y = alpha * std::exp(X);
      R = 1.0 / (beta + Y);
      T = 4 * U * U * V;
      T = std::log(T);
      T -= genCoef.u * X;
      T -= genCoef.s * std::log(genCoef.s * R);
    } while(T > 0);
    return Y * R;
  }

  /**
   * @fn variateAtkinsonWhittaker
   * Atkinson-Whittaker beta generator
   * @return beta variate for max(α, β) < 1 and α + β > 1
   */
  RealType variateAtkinsonWhittaker() const
  {
    size_t iter = 0;
    do
    {
      RealType U = UniformRand<RealType>::StandardVariate(this->localRandGenerator);
      RealType W = ExponentialRand<RealType>::StandardVariate(this->localRandGenerator);
      if(U <= genCoef.s)
      {
        RealType X = genCoef.t * std::pow(U / genCoef.s, 1.0 / alpha);
        if(W >= (1.0 - beta) * std::log((1.0 - X) / (1.0 - genCoef.t)))
          return X;
      }
      else
      {
        RealType X = 1.0 - (1.0 - genCoef.t) * std::pow((1.0 - U) / (1.0 - genCoef.s), 1.0 / beta);
        if(W >= (1.0 - alpha) * std::log(X / genCoef.t))
          return X;
      }
    } while(++iter <= this->MAX_ITER_REJECTION);
    throw std::runtime_error("Beta distribution: sampling failed");
  }

  /**
   * @fn variateGammaRatio
   * Gamma ratio beta generator
   * @return beta variate for the rest variations of shapes
   */
  RealType variateGammaRatio() const
  {
    RealType Y = GammaRV1.Variate();
    RealType Z = GammaRV2.Variate();
    return Y / (Y + Z);
  }
};

/**
 * @brief The BetaRand class <BR>
 * Beta distribution
 */
template <typename RealType = double>
class RANDLIB_EXPORT BetaRand : public BetaDistribution<RealType>, public ExponentialFamily<RealType, DoublePair>
{
public:
  BetaRand(double shape1 = 1, double shape2 = 1, double minValue = 0, double maxValue = 1)
  : BetaDistribution<RealType>(shape1, shape2, minValue, maxValue)
  {
  }

  String Name() const override
  {
    return "Beta(" + this->toStringWithPrecision(this->GetAlpha()) + ", " + this->toStringWithPrecision(this->GetBeta()) + ", " + this->toStringWithPrecision(this->MinValue()) + ", " +
           this->toStringWithPrecision(this->MaxValue()) + ")";
  }

  using BetaDistribution<RealType>::SetShapes;
  using BetaDistribution<RealType>::SetSupport;

  DoublePair SufficientStatistic(RealType x) const override
  {
    double y = (x - this->a) * this->bmaInv;
    return {std::log(y), std::log1pl(-y)};
  }

  DoublePair SourceParameters() const override
  {
    return {this->alpha, this->beta};
  }

  DoublePair SourceToNatural(DoublePair sourceParameters) const override
  {
    return {sourceParameters.first - 1, sourceParameters.second - 1};
  }

  double LogNormalizer(DoublePair theta) const override
  {
    return this->logbma + RandMath::logBeta(theta.first + 1, theta.second + 1);
  }

  DoublePair LogNormalizerGradient(DoublePair theta) const override
  {
    double psi1 = RandMath::digamma(theta.first + 1);
    double psi2 = RandMath::digamma(theta.second + 1);
    double psisum = RandMath::digamma(theta.first + theta.second + 2);
    return {psi1 - psisum, psi2 - psisum};
  }

  double CarrierMeasure(RealType) const override
  {
    return 0;
  }

  /**
   * @brief GetSampleLogMeanNorm
   * @param sample
   * @return mean average of log(x) for x from normalized sample
   */
  long double GetSampleLogMeanNorm(const std::vector<RealType>& sample) const
  {
    long double lnG = 0;
    for(RealType var : sample)
    {
      RealType x = (var - this->a) * this->bmaInv;
      lnG += std::log(x);
    }
    return lnG / sample.size();
  }

  /**
   * @brief GetSampleLog1pMeanNorm
   * @param sample
   * @return mean average of log(1+x) for x from normalized sample
   */
  long double GetSampleLog1pMeanNorm(const std::vector<RealType>& sample) const
  {
    long double lnG1p = 0;
    for(RealType var : sample)
    {
      RealType x = (var - this->a) * this->bmaInv;
      lnG1p += std::log1pl(x);
    }
    return lnG1p / sample.size();
  }

  /**
   * @brief GetSampleLog1mMeanNorm
   * @param sample
   * @return mean average of log(1-x) for x from normalized sample
   */
  long double GetSampleLog1mMeanNorm(const std::vector<RealType>& sample) const
  {
    long double lnG1m = 0;
    for(RealType var : sample)
    {
      RealType x = (var - this->a) * this->bmaInv;
      lnG1m += std::log1pl(-x);
    }
    return lnG1m / sample.size();
  }

  /**
   * @fn FitAlpha
   * set α, estimated via maximum likelihood,
   * using sufficient statistics instead of the whole sample
   * @param lnG normalized sample average of ln(X)
   * @param meanNorm normalized sample average
   */
  void FitAlpha(long double lnG, long double meanNorm)
  {
    if(meanNorm <= 0 || meanNorm >= 1)
      throw std::invalid_argument(this->fitErrorDescription(this->NOT_APPLICABLE, "Normalized mean of the sample should be in interval of (0, 1)"));
    if(this->beta == 1.0)
    {
      /// for β = 1 we have explicit expression for estimator
      SetShapes(-1.0 / lnG, this->beta);
    }
    else
    {
      /// get initial value for shape by method of moments
      RealType shape = meanNorm;
      shape /= 1.0 - meanNorm;
      shape *= this->beta;
      /// run root-finding procedure
      if(!RandMath::findRootNewtonFirstOrder<RealType>(
             [this, lnG](double x) {
               double first = RandMath::digamma(x) - RandMath::digamma(x + this->beta) - lnG;
               double second = RandMath::trigamma(x);
               return DoublePair(first, second);
             },
             shape))
        throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure."));
      SetShapes(shape, this->beta);
    }
  }

  /**
   * @fn FitAlpha
   * set α, estimated via maximum likelihood
   * @param sample
   */
  void FitAlpha(const std::vector<RealType>& sample)
  {
    if(!this->allElementsAreNotSmallerThan(this->a, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
    if(!this->allElementsAreNotGreaterThan(this->b, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));

    long double lnG = this->GetSampleLogMeanNorm(sample);
    if(!std::isfinite(lnG))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->ALPHA_ZERO));
    long double mean = this->GetSampleMean(sample);
    mean -= this->a;
    mean *= this->bmaInv;
    FitAlpha(lnG, mean);
  }

  /**
   * @fn FitBeta
   * set β, estimated via maximum likelihood,
   * using sufficient statistics instead of the whole sample
   * @param lnG1m normalized sample average of ln(1-X)
   * @param meanNorm normalized sample average
   */
  void FitBeta(long double lnG1m, long double meanNorm)
  {
    if(meanNorm <= 0 || meanNorm >= 1)
      throw std::invalid_argument(this->fitErrorDescription(this->NOT_APPLICABLE, "Normalized mean of the sample should be in interval of (0, 1)"));
    if(this->alpha == 1.0)
    {
      /// for α = 1 we have explicit expression for estimator
      SetShapes(this->alpha, -1.0 / lnG1m);
    }
    else
    {
      /// get initial value for shape by method of moments
      RealType shape = this->alpha / meanNorm - this->alpha;
      /// run root-finding procedure
      if(!RandMath::findRootNewtonFirstOrder<RealType>(
             [this, lnG1m](double x) {
               double first = RandMath::digamma(x) - RandMath::digamma(x + this->alpha) - lnG1m;
               double second = RandMath::trigamma(x);
               return DoublePair(first, second);
             },
             shape))
        throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure."));
      this->SetShapes(this->alpha, shape);
    }
  }

  /**
   * @fn FitBeta
   * set β, estimated via maximum likelihood
   * @param sample
   */
  void FitBeta(const std::vector<RealType>& sample)
  {
    if(!this->allElementsAreNotSmallerThan(this->a, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
    if(!this->allElementsAreNotGreaterThan(this->b, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));

    long double lnG1m = this->GetSampleLog1mMeanNorm(sample);
    if(!std::isfinite(lnG1m))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->BETA_ZERO));
    long double mean = this->GetSampleMean(sample);
    mean -= this->a;
    mean *= this->bmaInv;
    FitBeta(lnG1m, mean);
  }

  /**
   * @fn FitShapes
   * set α and β, estimated via maximum likelihood,
   * using sufficient statistics instead of the whole sample
   * @param lnG sample average of ln(X)
   * @param lnG1m sample average of ln(1-X)
   * @param mean sample average
   * @param variance sample variance
   */
  void FitShapes(long double lnG, long double lnG1m, long double mean, long double variance)
  {
    /// get initial values for shapes by method of moments
    double scaledMean = (mean - this->a) * this->bmaInv;
    double scaledVar = variance * this->bmaInv * this->bmaInv;
    double temp = scaledMean * (1.0 - scaledMean) / scaledVar - 1.0;
    double shape1 = 0.001, shape2 = shape1;
    if(temp > 0)
    {
      shape1 = scaledMean * temp;
      shape2 = (1.0 - scaledMean) * temp;
    }
    DoublePair shapes = std::make_pair(shape1, shape2);

    /// run root-finding procedure
    if(!RandMath::findRootNewtonFirstOrder2d(
           [lnG, lnG1m](DoublePair x) {
             double digammaAlphapBeta = RandMath::digamma(x.first + x.second);
             double digammaAlpha = RandMath::digamma(x.first);
             double digammaBeta = RandMath::digamma(x.second);
             double first = lnG + digammaAlphapBeta - digammaAlpha;
             double second = lnG1m + digammaAlphapBeta - digammaBeta;
             return DoublePair(first, second);
           },
           [](DoublePair x) {
             double trigammaAlphapBeta = RandMath::trigamma(x.first + x.second);
             double trigammaAlpha = RandMath::trigamma(x.first);
             double trigammaBeta = RandMath::trigamma(x.second);
             DoublePair first = std::make_pair(trigammaAlphapBeta - trigammaAlpha, trigammaAlphapBeta);
             DoublePair second = std::make_pair(trigammaAlphapBeta, trigammaAlphapBeta - trigammaBeta);
             return std::make_tuple(first, second);
           },
           shapes))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure."));
    SetShapes(shapes.first, shapes.second);
  }

  /**
   * @fn FitShapes
   * set α and β, estimated via maximum likelihood
   * @param sample
   */
  void FitShapes(const std::vector<RealType>& sample)
  {
    if(!this->allElementsAreNotSmallerThan(this->a, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
    if(!this->allElementsAreNotGreaterThan(this->b, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));

    long double lnG = this->GetSampleLogMeanNorm(sample);
    if(!std::isfinite(lnG))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->ALPHA_ZERO));
    long double lnG1m = this->GetSampleLog1mMeanNorm(sample);
    if(!std::isfinite(lnG1m))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->BETA_ZERO));

    /// get initial values for shapes by method of moments
    DoublePair stats = this->GetSampleMeanAndVariance(sample);
    FitShapes(lnG, lnG1m, stats.first, stats.second);
  }
};

/**
 * @brief The ArcsineRand class <BR>
 * Arcsine distribution
 *
 * Notation: X ~ Arcsine(α, a, b)
 *
 * Related distributions: <BR>
 * X ~ B(1 - α, α, a, b)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ArcsineRand : public BetaDistribution<RealType>
{
public:
  ArcsineRand(double shape = 0.5, double minValue = 0, double maxValue = 1)
  : BetaDistribution<RealType>(1.0 - shape, shape, minValue, maxValue)
  {
  }

  String Name() const override
  {
    return "Arcsine(" + this->toStringWithPrecision(GetShape()) + ", " + this->toStringWithPrecision(this->MinValue()) + ", " + this->toStringWithPrecision(this->MaxValue()) + ")";
  }

  using BetaDistribution<RealType>::SetSupport;

  void SetShape(double shape)
  {
    BetaDistribution<RealType>::SetShapes(1.0 - shape, shape);
  }

  inline double GetShape() const
  {
    return this->beta;
  }

  /**
   * @fn FitShape
   * set α and β, estimated via maximum likelihood,
   * using sufficient statistics instead of the whole sample
   * @param lnG average of all ln(X)
   * @param lnG1m average of all ln(1-X)
   */
  void FitShape(long double lnG, long double lnG1m)
  {
    double shape = M_PI / (lnG1m - lnG);
    if(!std::isfinite(shape))
      SetShape(0.5);
    shape = -M_1_PI * RandMath::atan(shape);
    SetShape(shape > 0 ? shape : shape + 1);
  }

  /**
   * @fn FitShape
   * set α, estimated via maximum likelihood
   * @param sample
   */
  void FitShape(const std::vector<RealType>& sample)
  {
    if(!this->allElementsAreNotSmallerThan(this->a, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
    if(!this->allElementsAreNotGreaterThan(this->b, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));

    int n = sample.size();
    long double lnG = 0, lnG1m = 0;
    for(double var : sample)
    {
      double x = (var - this->a) * this->bmaInv;
      lnG += std::log(x);
      lnG1m += std::log1pl(-x);
    }
    if(!std::isfinite(lnG))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->ALPHA_ZERO));
    if(!std::isfinite(lnG1m))
      throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, this->BETA_ZERO));
    lnG /= n;
    lnG1m /= n;
    FitShape(lnG, lnG1m);
  }
};

/**
 * @brief The BaldingNicholsRand class <BR>
 * Balding-Nichols distribution
 *
 * Notation: X ~ Balding-Nichols(F, p)
 *
 * Related distributions: <BR>
 * X ~ B(p * F', (1 - p) * F', 0, 1) for F' = (1 - F) / F
 */
template <typename RealType = double>
class RANDLIB_EXPORT BaldingNicholsRand : public BetaDistribution<RealType>
{
  double F = 0.5, p = 0.5;

public:
  BaldingNicholsRand(double fixatingIndex, double frequency)
  {
    SetFixatingIndexAndFrequency(fixatingIndex, frequency);
  }

  String Name() const override
  {
    return "Balding-Nichols(" + this->toStringWithPrecision(GetFixatingIndex()) + ", " + this->toStringWithPrecision(GetFrequency()) + ")";
  }

  void SetFixatingIndexAndFrequency(double fixatingIndex, double frequency)
  {
    F = fixatingIndex;
    if(F <= 0 || F >= 1)
      F = 0.5;

    p = frequency;
    if(p <= 0 || p >= 1)
      p = 0.5;

    double frac = (1.0 - F) / F, fracP = frac * p;
    BetaDistribution<RealType>::SetShapes(fracP, frac - fracP);
  }

  inline double GetFixatingIndex() const
  {
    return F;
  }

  inline double GetFrequency() const
  {
    return p;
  }
};
} // namespace randlib
