#pragma once

#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

#include "distributions/univariate/ExponentialFamily.hpp"

namespace randlib
{
/**
 * @brief The GammaDistribution class <BR>
 * Abstract class for Gamma distribution
 *
 * f(x | α, β) = β^α / Γ(α) * x^(α-1) * exp(-βx),
 * where Γ(α) denotes Gamma function
 *
 * Notation X ~ Γ(α, β)
 *
 * Related distributions: <BR>
 * σX ~ Γ(α, σβ) <BR>
 * If X ~ Γ(1, β), then X ~ Exp(β) <BR>
 * If X ~ Γ(0.5 * n, 0.5), then X ~ χ^2(n) <BR>
 * If X ~ Γ(k, β) for integer k, then X ~ Erlang(k, β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT GammaDistribution : virtual public randlib::ContinuousDistribution<RealType>
{
protected:
  double alpha = 1;               ///< shape α
  double beta = 1;                ///< rate β
  double theta = 1;               ///< scale θ = 1/β
  double lgammaAlpha = 0;         ///< log(Γ(α))
  double pdfCoef = 0;             ///< α * log(β) - log(Γ(α))
  double logAlpha = 0;            ///< log(α)
  double logBeta = 0;             ///< log(β)
  double digammaAlpha = -M_EULER; ///< ψ(α)

protected:
  GammaDistribution(double shape, double rate)
  {
    SetParameters(shape, rate);
  }

  virtual ~GammaDistribution()
  {
  }

public:
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

  /**
   * @fn GetShape
   * @return α
   */
  inline double GetShape() const
  {
    return alpha;
  }

  /**
   * @fn GetScale
   * @return θ = 1/β
   */
  inline double GetScale() const
  {
    return theta;
  }

  /**
   * @fn GetRate
   * @return β
   */
  inline double GetRate() const
  {
    return beta;
  }

  /**
   * @fn GetLogGammaShape
   * @return log(Γ(α))
   */
  inline double GetLogGammaShape() const
  {
    return lgammaAlpha;
  }

  /**
   * @fn GetLogShape
   * @return log(α)
   */
  inline double GetLogShape() const
  {
    return logAlpha;
  }

  /**
   * @fn GetDigammaShape
   * @return ψ(α)
   */
  inline double GetDigammaShape() const
  {
    return digammaAlpha;
  }

  /**
   * @fn GetLogRate
   * @return log(β)
   */
  inline double GetLogRate() const
  {
    return logBeta;
  }

  double f(const RealType& x) const override
  {
    if(x < 0.0)
      return 0.0;
    if(x == 0.0)
    {
      if(this->alpha > 1.0)
        return 0.0;
      return (this->alpha == 1.0) ? this->beta : INFINITY;
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0.0)
    {
      if(this->alpha > 1.0)
        return -INFINITY;
      return (this->alpha == 1.0) ? this->logBeta : INFINITY;
    }
    double y = (this->alpha - 1.0) * std::log(x);
    y -= x * this->beta;
    y += pdfCoef;
    return y;
  }

  double F(const RealType& x) const override
  {
    return (x > 0.0) ? RandMath::pgamma(this->alpha, x * this->beta, logAlpha, lgammaAlpha) : 0.0;
  }

  double logF(const RealType& x) const
  {
    return (x > 0.0) ? RandMath::lpgamma(this->alpha, x * this->beta, logAlpha, lgammaAlpha) : -INFINITY;
  }

  double S(const RealType& x) const override
  {
    return (x > 0.0) ? RandMath::qgamma(this->alpha, x * this->beta, logAlpha, lgammaAlpha) : 1.0;
  }

  double logS(const RealType& x) const
  {
    return (x > 0.0) ? RandMath::lqgamma(this->alpha, x * this->beta, logAlpha, lgammaAlpha) : 0.0;
  }

  /**
   * @fn StandardVariate
   * @param shape α
   * @return gamma variate with shape α and unity rate
   */
  static RealType StandardVariate(double shape, RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    if(shape <= 0)
      throw std::invalid_argument("Gamma distribution: shape should be positive");

    GENERATOR_ID genId = getIdOfUsedGenerator(shape);

    switch(genId)
    {
    case INTEGER_SHAPE:
      return variateThroughExponentialSum(std::round(shape), randGenerator);
    case ONE_AND_A_HALF_SHAPE:
      return variateForShapeOneAndAHalf(randGenerator);
    case SMALL_SHAPE:
      return variateAhrensDieter(shape, randGenerator);
    case FISHMAN:
      return variateFishman(shape, randGenerator);
    case MARSAGLIA_TSANG:
      return variateMarsagliaTsang(shape, randGenerator);
    default:
      throw std::runtime_error("Gamma distribution: invalid generator id");
    }
  }

  /**
   * @fn Variate
   * @param shape α
   * @param rate β
   * @return gamma variate with shape α and rate β
   */
  static RealType Variate(double shape, double rate, RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Gamma distribution: shape should be positive");
    if(rate <= 0.0)
      throw std::invalid_argument("Gamma distribution: rate should be positive");
    return StandardVariate(shape, randGenerator) / rate;
  }

  RealType Variate() const override
  {
    GENERATOR_ID genId = getIdOfUsedGenerator(this->alpha);

    switch(genId)
    {
    case INTEGER_SHAPE:
      return theta * variateThroughExponentialSum(this->alpha, this->localRandGenerator);
    case ONE_AND_A_HALF_SHAPE:
      return theta * variateForShapeOneAndAHalf(this->localRandGenerator);
    case SMALL_SHAPE:
      return theta * variateBest(this->localRandGenerator);
    case FISHMAN:
      return theta * variateFishman(this->alpha, this->localRandGenerator);
    case MARSAGLIA_TSANG:
      return theta * variateMarsagliaTsang(this->alpha, this->localRandGenerator);
    default:
      throw std::runtime_error("Gamma distribution: invalid generator id");
    }
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    GENERATOR_ID genId = getIdOfUsedGenerator(this->alpha);

    switch(genId)
    {
    case INTEGER_SHAPE:
      for(RealType& var : outputData)
        var = theta * variateThroughExponentialSum(this->alpha, this->localRandGenerator);
      break;
    case ONE_AND_A_HALF_SHAPE:
      for(RealType& var : outputData)
        var = theta * variateForShapeOneAndAHalf(this->localRandGenerator);
      break;
    case SMALL_SHAPE:
      for(RealType& var : outputData)
        var = theta * variateBest(this->localRandGenerator);
      break;
    case FISHMAN:
      for(RealType& var : outputData)
        var = theta * variateFishman(this->alpha, this->localRandGenerator);
      break;
    case MARSAGLIA_TSANG:
      for(RealType& var : outputData)
        var = theta * variateMarsagliaTsang(this->alpha, this->localRandGenerator);
      break;
    default:
      return;
    }
  }

  /**
   * @fn Mean
   * @return E[X]
   */
  long double Mean() const override
  {
    return this->alpha * theta;
  }

  /**
   * @fn GeometricMean
   * @return E[ln(X)]
   */
  long double GeometricMean() const
  {
    return this->digammaAlpha - this->logBeta;
  }

  /**
   * @fn Variance
   * @return Var(X)
   */
  long double Variance() const override
  {
    return this->alpha * theta * theta;
  }

  /**
   * @fn GeometricVariance
   * @return Var(ln(X))
   */
  long double GeometricVariance() const
  {
    return RandMath::trigamma(this->alpha);
  }

  RealType Mode() const override
  {
    return (this->alpha <= 1) ? 0 : (this->alpha - 1) * theta;
  }

  RealType Median() const override
  {
    return (this->alpha == 1.0) ? theta * M_LN2 : quantileImpl(0.5);
  }

  long double Skewness() const override
  {
    return 2.0l / std::sqrt(this->alpha);
  }

  long double ExcessKurtosis() const override
  {
    return 6.0l / this->alpha;
  }

protected:
  /**
   * @fn SetParameters
   * @param shape α
   * @param rate β
   */
  void SetParameters(double shape, double rate)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Gamma distribution: shape should be positive, but it's equal to " + std::to_string(shape));
    if(rate <= 0.0)
      throw std::invalid_argument("Gamma distribution: rate should be positive, but it's equal to " + std::to_string(rate));

    this->alpha = shape > 0 ? shape : 1.0;

    this->beta = (rate > 0.0) ? rate : 1.0;
    theta = 1.0 / this->beta;

    lgammaAlpha = std::lgammal(this->alpha);
    logAlpha = std::log(this->alpha);
    this->logBeta = std::log(this->beta);
    pdfCoef = -lgammaAlpha + this->alpha * this->logBeta;

    if(getIdOfUsedGenerator(this->alpha) == SMALL_SHAPE)
    {
      /// set constants for generator
      genCoef.t = 0.5 * std::log1pl(-alpha);
      genCoef.t = 0.07 + 0.75 * std::exp(genCoef.t);
      genCoef.b = 1.0 + std::exp(-genCoef.t) * this->alpha / genCoef.t;
    }
  }

  /**
   * @brief SetShape
   * @param shape α
   */
  void SetShape(double shape)
  {
    SetParameters(shape, this->beta);
  }

private:
  /// constants for faster sampling
  struct genCoef_t
  {
    double t, b;
  } genCoef = {0, 0};

  enum GENERATOR_ID
  {
    INTEGER_SHAPE,        ///< Erlang distribution for α = 1, 2, 3
    ONE_AND_A_HALF_SHAPE, ///< α = 1.5
    SMALL_SHAPE,          ///< α < 0.34
    FISHMAN,              ///< 1 < α < 1.2
    MARSAGLIA_TSANG       ///< 0.34 < α < 1 or α >= 1.2
  };

  /**
   * @fn getIdOfUsedGenerator
   * @param shape α
   * @return id of used variate generator according to the shape
   */
  static GENERATOR_ID getIdOfUsedGenerator(double shape)
  {
    if(shape < 0.34)
      return SMALL_SHAPE;
    if(shape <= 3.0 && RandMath::areClose(shape, std::round(shape)))
      return INTEGER_SHAPE;
    if(RandMath::areClose(shape, 1.5))
      return ONE_AND_A_HALF_SHAPE;
    if(shape > 1.0 && shape < 1.2)
      return FISHMAN;
    return MARSAGLIA_TSANG;
  }

  /**
   * @fn variateThroughExponentialSum
   * @param shape α
   * @return gamma variate, generated by sum of exponentially distributed random
   * variables
   */
  static RealType variateThroughExponentialSum(int shape, RandGenerator& randGenerator)
  {
    double X = 0.0;
    for(int i = 0; i < shape; ++i)
      X += ExponentialRand<RealType>::StandardVariate(randGenerator);
    return X;
  }

  /**
   * @fn variateForShapeOneAndAHalf
   * @return gamma variate for α = 1.5
   */
  static RealType variateForShapeOneAndAHalf(RandGenerator& randGenerator)
  {
    RealType W = ExponentialRand<RealType>::StandardVariate(randGenerator);
    RealType N = NormalRand<RealType>::StandardVariate(randGenerator);
    return W + 0.5 * N * N;
  }

  /**
   * @fn variateBest
   * @return gamma variate for small α, using Best algorithm
   */
  RealType variateBest(RandGenerator& randGenerator) const
  {
    /// Algorithm RGS for gamma variates (Best, 1983)
    double X = 0;
    size_t iter = 0;
    do
    {
      double V = genCoef.b * UniformRand<RealType>::StandardVariate(randGenerator);
      double W = UniformRand<RealType>::StandardVariate(randGenerator);
      if(V <= 1)
      {
        X = genCoef.t * std::pow(V, 1.0 / this->alpha);
        if(W <= (2.0 - X) / (2.0 + X) || W <= std::exp(-X))
          return X;
      }
      else
      {
        X = -std::log(genCoef.t * (genCoef.b - V) / this->alpha);
        double Y = X / genCoef.t;
        if(W * (this->alpha + Y - this->alpha * Y) <= 1 || W <= std::pow(Y, this->alpha - 1))
          return X;
      }
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Gamma distribution: sampling failed");
  }

  /**
   * @fn variateAhrensDieter
   * @param shape α
   * @return gamma variate for small α, using Ahrens-Dieter algorithm (if we
   * have no pre-calculated values)
   */
  static RealType variateAhrensDieter(double shape, RandGenerator& randGenerator)
  {
    /// Rejection algorithm GS for gamma variates (Ahrens and Dieter, 1974)
    double X = 0;
    size_t iter = 0;
    double shapeInv = 1.0 / shape;
    double t = shapeInv + M_1_E;
    do
    {
      double U = UniformRand<RealType>::StandardVariate(randGenerator);
      double p = shape * t * U;
      double W = ExponentialRand<RealType>::StandardVariate(randGenerator);
      if(p <= 1)
      {
        X = std::pow(p, shapeInv);
        if(X <= W)
          return X;
      }
      else
      {
        X = -std::log(t * (1 - U));
        if((1 - shape) * std::log(X) <= W)
          return X;
      }
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Gamma distribution: sampling failed");
  }

  /**
   * @fn variateFishman
   * @param shape α
   * @return gamma variate, using Fishman algorithm
   */
  static RealType variateFishman(double shape, RandGenerator& randGenerator)
  {
    /// G. Fishman algorithm (shape > 1)
    double W1, W2;
    double shapem1 = shape - 1;
    do
    {
      W1 = ExponentialRand<RealType>::StandardVariate(randGenerator);
      W2 = ExponentialRand<RealType>::StandardVariate(randGenerator);
    } while(W2 < shapem1 * (W1 - std::log(W1) - 1));
    return shape * W1;
  }

  /**
   * @fn variateMarsagliaTsang
   * @param shape α
   * @return gamma variate, using Marsaglia-Tsang algorithm
   */
  static RealType variateMarsagliaTsang(double shape, RandGenerator& randGenerator)
  {
    /// Marsaglia and Tsang’s Method (shape > 1/3)
    RealType d = shape - 1.0 / 3;
    RealType c = 3 * std::sqrt(d);
    size_t iter = 0;
    do
    {
      RealType N;
      do
      {
        N = NormalRand<RealType>::StandardVariate(randGenerator);
      } while(N <= -c);
      RealType v = 1 + N / c;
      v = v * v * v;
      N *= N;
      RealType U = UniformRand<RealType>::StandardVariate(randGenerator);
      if(U < 1.0 - 0.331 * N * N || std::log(U) < 0.5 * N + d * (1.0 - v + std::log(v)))
      {
        return d * v;
      }
    } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Gamma distribution: sampling failed");
  }

  /// quantile auxiliary functions
  RealType initRootForSmallP(double r) const
  {
    double root = 0;
    double c[5];
    c[4] = 1;
    /// first coefficient
    double denominator = this->alpha + 1;
    c[3] = 1.0 / denominator;
    /// second coefficient
    denominator *= denominator;
    denominator *= this->alpha + 2;
    c[2] = 0.5 * (3 * this->alpha + 5) / denominator;
    /// third coefficient
    denominator *= 3 * (this->alpha + 1) * (this->alpha + 3);
    c[1] = 8 * this->alpha + 33;
    c[1] *= this->alpha;
    c[1] += 31;
    c[1] /= denominator;
    /// fourth coefficient
    denominator *= 8 * (this->alpha + 1) * (this->alpha + 2) * (this->alpha + 4);
    c[0] = 125 * this->alpha + 1179;
    c[0] *= this->alpha;
    c[0] += 3971;
    c[0] *= this->alpha;
    c[0] += 5661;
    c[0] *= this->alpha;
    c[0] += 2888;
    c[0] /= denominator;
    /// now calculate root
    for(int i = 0; i != 5; ++i)
    {
      root += c[i];
      root *= r;
    }
    return root;
  }

  RealType initRootForLargeP(double logQ) const
  {
    /// look for approximate value of x -> INFINITY
    double x = (logQ + lgammaAlpha) / this->alpha;
    x = -std::exp(x) / this->alpha;
    return -alpha * RandMath::Wm1Lambert(x);
  }

  RealType initRootForSmallShape(double p) const
  {
    throw std::runtime_error("initRootForSmallShape - This function has not been implemented!");
  }

  RealType initRootForLargeShape(double p) const
  {
    if(p == 0.5)
      return this->alpha;
    double x = RandMath::erfcinv(2 * p);
    double lambda = x * x / this->alpha + 1;
    lambda = -std::exp(-lambda);
    lambda = (x < 0) ? -RandMath::Wm1Lambert(lambda) : -RandMath::W0Lambert(lambda);
    return lambda * this->alpha;
  }

  RealType initRootForLargeShape1m(double p) const
  {
    if(p == 0.5)
      return this->alpha;
    double x = -RandMath::erfcinv(2 * p);
    double lambda = x * x / this->alpha + 1;
    lambda = -std::exp(-lambda);
    lambda = (x < 0) ? -RandMath::Wm1Lambert(lambda) : -RandMath::W0Lambert(lambda);
    return lambda * this->alpha;
  }

  RealType quantileInitialGuess(double p) const
  {
    /// Method is taken from
    /// "Efficient and accurate algorithms
    /// for the computation and inversion
    /// of the incomplete gamma function ratios"
    /// (Amparo Gil, Javier Segura and Nico M. Temme)
    double guess = 0;
    if(this->alpha < 10)
    {
      double r = std::log(p * this->alpha) + lgammaAlpha;
      r = std::exp(r / this->alpha);
      /// if p -> 0
      if(r < 0.2 * (this->alpha + 1))
      {
        guess = initRootForSmallP(r);
      }
      else
      {
        double logQ = std::log1pl(-p);
        /// boundary adviced in a paper
        double maxBoundary1 = -0.5 * this->alpha - logAlpha - lgammaAlpha;
        /// the maximum possible value to have a solution
        double maxBoundary2 = this->alpha * (logAlpha - 1) - lgammaAlpha;
        /// if p -> 1
        if(logQ < std::min(maxBoundary1, maxBoundary2))
          guess = initRootForLargeP(logQ);
        else if(this->alpha < 1)
          guess = r;
        else
          guess = initRootForLargeShape(p);
      }
    }
    else
      guess = initRootForLargeShape(p);
    return guess / this->beta;
  }

  RealType quantileInitialGuess1m(double p) const
  {
    if(this->alpha < 10)
    {
      double logQ = std::log(p);
      /// boundary adviced in a paper
      double maxBoundary1 = -0.5 * this->alpha - logAlpha - lgammaAlpha;
      /// the maximum possible value to have a solution
      double maxBoundary2 = this->alpha * (logAlpha - 1) - lgammaAlpha;
      /// if p -> 0
      if(logQ < std::min(maxBoundary1, maxBoundary2))
        return initRootForLargeP(logQ) / this->beta;
    }
    else
    {
      return initRootForLargeShape1m(p) / this->beta;
    }
    return quantileInitialGuess(1.0 - p);
  }

  /**
   * @fn df
   * derivative of probability density function
   * @param x
   * @return f'(x)
   */
  double df(RealType x) const
  {
    double z = (this->alpha - 1) - this->beta * x;
    double y = (this->alpha - 2) * std::log(x);
    y -= this->beta * x;
    y += pdfCoef;
    return z * std::exp(y);
  }

  /**
   * @fn dfDivf
   * derivative of pdf, divided by pdf
   * @param x
   * @return f'(x) / f(x)
   */
  double dfDivf(RealType x) const
  {
    return x / (this->alpha - 1 - this->beta * x);
  }

  RealType quantileImpl(double p, RealType initValue) const override
  {
    if(p < 1e-5)
    { /// too small p
      double logP = std::log(p);
      if(!RandMath::findRootNewtonSecondOrder<RealType>(
             [this, logP](double x) {
               if(x <= 0)
                 return DoubleTriplet(-INFINITY, 0, 0);
               double logCdf = logF(x), logPdf = logf(x);
               double first = logCdf - logP;
               double second = std::exp(logPdf - logCdf);
               double third = second * (dfDivf(x) - second);
               return DoubleTriplet(first, second, third);
             },
             initValue))
        throw std::runtime_error("Gamma distribution: failure in numeric procedure");
      return initValue;
    }
    if(!RandMath::findRootNewtonSecondOrder<RealType>(
           [this, p](double x) {
             if(x <= 0)
               return DoubleTriplet(-p, 0, 0);
             double first = F(x) - p;
             double second = f(x);
             double third = df(x);
             return DoubleTriplet(first, second, third);
           },
           initValue))
      throw std::runtime_error("Gamma distribution: failure in numeric procedure");
    return initValue;
  }

  RealType quantileImpl(double p) const override
  {
    return (this->alpha == 1.0) ? -theta * std::log1pl(-p) : quantileImpl(p, quantileInitialGuess(p));
  }

  RealType quantileImpl1m(double p, RealType initValue) const override
  {
    if(p < 1e-5)
    { /// too small p
      double logP = std::log(p);
      if(!RandMath::findRootNewtonSecondOrder<RealType>(
             [this, logP](double x) {
               if(x <= 0)
                 return DoubleTriplet(logP, 0, 0);
               double logCcdf = logS(x), logPdf = logf(x);
               double first = logP - logCcdf;
               double second = std::exp(logPdf - logCcdf);
               double third = second * (dfDivf(x) + second);
               return DoubleTriplet(first, second, third);
             },
             initValue))
        throw std::runtime_error("Gamma distribution: failure in numeric procedure");
      return initValue;
    }
    if(!RandMath::findRootNewtonSecondOrder<RealType>(
           [this, p](double x) {
             if(x <= 0)
               return DoubleTriplet(p - 1.0, 0, 0);
             double first = p - S(x);
             double second = f(x);
             double third = df(x);
             return DoubleTriplet(first, second, third);
           },
           initValue))
      throw std::runtime_error("Gamma distribution: failure in numeric procedure");
    return initValue;
  }

  RealType quantileImpl1m(double p) const override
  {
    return (this->alpha == 1.0) ? -theta * std::log(p) : quantileImpl1m(p, quantileInitialGuess1m(p));
  }

  std::complex<double> CFImpl(double t) const override
  {
    return std::pow(std::complex<double>(1.0, -theta * t), -alpha);
  }
};

/**
 * @brief The FreeRateGammaDistribution class <BR>
 * Abstract class for Gamma distribution with arbitrary scale/rate
 */
template <typename RealType = double>
class RANDLIB_EXPORT FreeRateGammaDistribution : public GammaDistribution<RealType>
{
protected:
  FreeRateGammaDistribution(double shape, double rate)
  : GammaDistribution<RealType>(shape, rate)
  {
  }

public:
  /**
   * @fn SetRate
   * set rate β
   * @param rate
   */
  void SetRate(double rate)
  {
    this->SetParameters(this->alpha, rate);
  }

  /**
   * @fn SetScale
   * set scale θ = 1/β
   * @param scale
   */
  void SetScale(double scale)
  {
    SetRate(1.0 / scale);
  }

  /**
   * @fn FitRate
   * set rate, estimated via maximum-likelihood method if unbiased = false,
   * otherwise set rate, returned by uniformly minimum variance unbiased
   * estimator
   * @param sample
   */
  void FitRate(const std::vector<RealType>& sample, bool unbiased = false)
  {
    /// Sanity check
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    double mean = this->GetSampleMean(sample);
    double coef = this->alpha - (unbiased ? 1.0 / sample.size() : 0.0);
    this->SetParameters(this->alpha, coef / mean);
  }

  /**
   * @fn FitRateBayes
   * set rate, returned by bayesian estimation
   * @param sample
   * @param priorDistribution
   * @return posterior distribution
   */
  GammaRand<RealType> FitRateBayes(const std::vector<RealType>& sample, const GammaDistribution<RealType>& priorDistribution, bool MAP = false)
  {
    /// Sanity check
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    double kappa = priorDistribution.GetShape();
    double gamma = priorDistribution.GetRate();
    double newShape = this->alpha * sample.size() + kappa;
    double newRate = this->GetSampleSum(sample) + gamma;
    GammaRand<RealType> posteriorDistribution(newShape, newRate);
    this->SetParameters(this->alpha, MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }
};

/**
 * @brief The GammaRand class <BR>
 * Gamma distribution
 */
template <typename RealType = double>
class RANDLIB_EXPORT GammaRand : public FreeRateGammaDistribution<RealType>, public ExponentialFamily<RealType, DoublePair>
{
public:
  GammaRand(double shape = 1, double rate = 1)
  : FreeRateGammaDistribution<RealType>(shape, rate)
  {
  }

  String Name() const override
  {
    return "Gamma(" + this->toStringWithPrecision(this->GetShape()) + ", " + this->toStringWithPrecision(this->GetRate()) + ")";
  }

  using GammaDistribution<RealType>::SetParameters;
  using GammaDistribution<RealType>::SetShape;

  /**
   * @fn FitShape
   * set shape, estimated via maximum-likelihood method
   * @param sample
   */
  void FitShape(const std::vector<RealType>& sample)
  {
    /// Sanity check
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));

    /// Calculate initial guess via method of moments
    double shape = this->GetSampleMean(sample) * this->beta;
    /// Run root-finding procedure
    double s = this->GetSampleLogMean(sample) + this->logBeta;
    if(!RandMath::findRootNewtonFirstOrder<double>(
           [s](double x) {
             double first = RandMath::digamma(x) - s;
             double second = RandMath::trigamma(x);
             return DoublePair(first, second);
           },
           shape))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure"));
    SetParameters(shape, this->beta);
  }

  /**
   * @fn Fit
   * set shape and rate, estimated via maximum-likelihood method
   * @param sample
   */
  void Fit(const std::vector<RealType>& sample)
  {
    /// Sanity check
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));

    /// Calculate initial guess for shape
    double average = this->GetSampleMean(sample);
    double s = std::log(average) - this->GetSampleLogMean(sample);
    double sm3 = s - 3.0, sp12 = 12.0 * s;
    double shape = sm3 * sm3 + 2 * sp12;
    shape = std::sqrt(shape);
    shape -= sm3;
    shape /= sp12;

    if(!RandMath::findRootNewtonFirstOrder<double>(
           [s](double x) {
             double first = RandMath::digammamLog(x) + s;
             double second = RandMath::trigamma(x) - 1.0 / x;
             return DoublePair(first, second);
           },
           shape))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure"));

    SetParameters(shape, shape / average);
  }

  DoublePair SufficientStatistic(RealType x) const override
  {
    return {std::log(x), x};
  }

  DoublePair SourceParameters() const override
  {
    return {this->alpha, this->beta};
  }

  DoublePair SourceToNatural(DoublePair sourceParameters) const override
  {
    double shape = sourceParameters.first;
    double rate = sourceParameters.second;
    return {shape - 1, -rate};
  }

  double LogNormalizer(DoublePair parameters) const override
  {
    double shape = parameters.first + 1, rate = -parameters.second;
    double F = std::lgamma(shape);
    F -= shape * std::log(rate);
    return F;
  }

  DoublePair LogNormalizerGradient(DoublePair parameters) const override
  {
    double shape = parameters.first + 1, rate = -parameters.second;
    double gradF1 = RandMath::digamma(shape) - std::log(rate);
    double gradF2 = shape / rate;
    return {gradF1, gradF2};
  }

  double CarrierMeasure(RealType) const override
  {
    return 0;
  }

  double CrossEntropyAdjusted(DoublePair parameters) const override
  {
    double shapeq = parameters.first + 1, rateq = -parameters.second;
    double H = std::lgamma(shapeq);
    H -= shapeq * std::log(rateq);
    H -= (shapeq - 1) * (this->digammaAlpha - this->logBeta);
    H += rateq * this->alpha / this->beta;
    return H;
  }

  double EntropyAdjusted() const override
  {
    double H = this->lgammaAlpha;
    H -= this->logBeta;
    H -= (this->alpha - 1) * this->digammaAlpha;
    H += this->alpha;
    return H;
  }
};

/**
 * @brief The ChiSquaredRand class <BR>
 * Chi-squared distribution
 *
 * Notation: X ~ χ^2(k)
 *
 * Related distributions: <BR>
 * X ~ Γ(0.5 * k, 0.5)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ChiSquaredRand : public GammaDistribution<RealType>
{
public:
  explicit ChiSquaredRand(size_t degree = 1)
  : GammaDistribution<RealType>(0.5 * degree, 0.5)
  {
  }

  String Name() const override
  {
    return "Chi-squared(" + this->toStringWithPrecision(GetDegree()) + ")";
  }

  void SetDegree(size_t degree)
  {
    GammaDistribution<RealType>::SetParameters(0.5 * degree, 0.5);
  }

  inline size_t GetDegree() const
  {
    return static_cast<int>(2 * this->alpha);
  }
};

/**
 * @brief The ErlangRand class <BR>
 * Erlang distibution
 *
 * Notation: X ~ Erlang(k, β)
 *
 * Related distributions: <BR>
 * X ~ Y_1 + Y_2 + ... + Y_k, where Y_i ~ Exp(β) <BR>
 * X ~ Γ(k, β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ErlangRand : public FreeRateGammaDistribution<RealType>
{
public:
  ErlangRand(int shape = 1, double rate = 1)
  : FreeRateGammaDistribution<RealType>(shape, rate)
  {
  }

  String Name() const override
  {
    return "Erlang(" + this->toStringWithPrecision(this->GetShape()) + ", " + this->toStringWithPrecision(this->GetRate()) + ")";
  }

  void SetParameters(size_t shape, double rate)
  {
    GammaDistribution<RealType>::SetParameters(shape, rate);
  }

  void SetShape(size_t shape)
  {
    SetParameters(shape, this->beta);
  }
};
} // namespace randlib