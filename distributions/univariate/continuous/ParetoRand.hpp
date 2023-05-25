#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"

namespace randlib
{
/**
 * @brief The ParetoRand class <BR>
 * Pareto distribution
 *
 * Notation: X ~ Pareto(α, σ)
 *
 * Related distributions: <BR>
 * ln(X / σ) ~ Exp(α)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ParetoRand : public randlib::ContinuousDistribution<RealType>
{
  double alpha = 1;    ///< shape α
  double sigma = 1;    ///< scale σ
  double logAlpha = 0; ///< log(α)
  double logSigma = 0; ///< log(σ)

public:
  ParetoRand(double shape = 1, double scale = 1)
  {
    SetShape(shape);
    SetScale(scale);
  }

  String Name() const override
  {
    return "Pareto(" + this->toStringWithPrecision(GetShape()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  RealType MinValue() const override
  {
    return sigma;
  }

  RealType MaxValue() const override
  {
    return INFINITY;
  }

  void SetShape(double shape)
  {
    if(shape <= 0.0)
      throw std::invalid_argument("Pareto distribution: shape should be positive");
    alpha = shape;
    logAlpha = std::log(alpha);
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Pareto distribution: scale should be positive");
    sigma = scale;
    logSigma = std::log(sigma);
  }

  inline double GetShape() const
  {
    return alpha;
  }

  inline double GetScale() const
  {
    return sigma;
  }

  inline double GetLogShape() const
  {
    return logAlpha;
  }

  inline double GetLogScale() const
  {
    return logSigma;
  }

  double f(const RealType& x) const override
  {
    return (x < sigma) ? 0.0 : std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < sigma)
      return -INFINITY;
    double logX = std::log(x);
    double y = logSigma - logX;
    y *= alpha;
    y -= logX;
    y += logAlpha;
    return y;
  }

  double F(const RealType& x) const override
  {
    return (x > sigma) ? -std::expm1l(alpha * std::log(sigma / x)) : 0.0;
  }

  double S(const RealType& x) const override
  {
    return (x > sigma) ? std::pow(sigma / x, alpha) : 1.0;
  }

  RealType Variate() const override
  {
    return sigma * StandardVariate(alpha, this->localRandGenerator);
  }

  static RealType StandardVariate(double shape, RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    if(RandMath::areClose(shape, 1.0))
      return variateForAlphaEqualOne(randGenerator);
    if(RandMath::areClose(shape, 2.0))
      return variateForAlphaEqualTwo(randGenerator);
    return variateForGeneralAlpha(shape, randGenerator);
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    if(RandMath::areClose(alpha, 1.0))
    {
      for(RealType& var : outputData)
        var = sigma * variateForAlphaEqualOne(this->localRandGenerator);
    }
    else if(RandMath::areClose(alpha, 2.0))
    {
      for(RealType& var : outputData)
        var = sigma * variateForAlphaEqualTwo(this->localRandGenerator);
    }
    else
    {
      for(RealType& var : outputData)
        var = sigma * variateForGeneralAlpha(alpha, this->localRandGenerator);
    }
  }

  long double Mean() const override
  {
    return (alpha > 1) ? alpha * sigma / (alpha - 1) : INFINITY;
  }

  long double Variance() const override
  {
    if(alpha > 2)
    {
      long double var = sigma / (alpha - 1);
      var *= var;
      return alpha * var / (alpha - 2);
    }
    return (alpha > 1) ? INFINITY : NAN;
  }

  RealType Median() const override
  {
    return std::exp(logSigma + M_LN2 / alpha);
  }

  RealType Mode() const override
  {
    return sigma;
  }

  long double Skewness() const override
  {
    if(alpha <= 3)
      return INFINITY;
    double skewness = (alpha - 2.0) / alpha;
    skewness = std::sqrt(skewness);
    skewness *= (1 + alpha) / (alpha - 3);
    return skewness + skewness;
  }

  long double ExcessKurtosis() const override
  {
    if(alpha <= 4)
      return INFINITY;
    double numerator = alpha + 1;
    numerator *= alpha;
    numerator -= 6;
    numerator *= alpha;
    numerator -= 2;
    double denominator = alpha * (alpha - 3) * (alpha - 4);
    return 6.0 * numerator / denominator;
  }

  long double Entropy() const
  {
    return logSigma - logAlpha + 1.0 / alpha + 1;
  }

  /**
   * @fn FitShape
   * Fit α to maximum-likelihood estimator
   * or to UMVU if unbiased == true
   * @param sample
   * @param unbiased
   */
  void FitShape(const std::vector<RealType>& sample, bool unbiased = false)
  {
    if(!this->allElementsAreNotSmallerThan(sigma, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(sigma)));
    double invShape = this->GetSampleLogMean(sample) - logSigma;
    if(invShape == 0.0)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, "Possibly all the elements of the sample coincide "
                                                                                "with the lower boundary σ."));
    double shape = 1.0 / invShape;
    if(unbiased)
      shape *= (1.0 - 1.0 / sample.size());
    SetShape(shape);
  }

  /**
   * @fn FitScale
   * Fit σ to maximum-likelihood estimator
   * or to UMVU if unbiased == true
   * @param sample
   * @param unbiased
   */
  void FitScale(const std::vector<RealType>& sample, bool unbiased = false)
  {
    RealType minVar = *std::min_element(sample.begin(), sample.end());
    if(minVar <= 0)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    double scale = unbiased ? (1.0 - 1.0 / (sample.size() * alpha)) * minVar : minVar;
    SetScale(scale);
  }

  /**
   * @fn Fit
   * Fit parameters to maximum-likelihood estimators
   * or to UMVU if unbiased == true
   * @param sample
   * @param unbiased
   */
  void Fit(const std::vector<RealType>& sample, bool unbiased = false)
  {
    RealType minVar = *std::min_element(sample.begin(), sample.end());
    if(minVar <= 0)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));

    double logBiasedSigma = std::log(minVar);
    double invShape = this->GetSampleLogMean(sample) - logBiasedSigma;
    if(invShape == 0.0)
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, "Possibly all the elements of the sample are the same."));
    double shape = 1.0 / invShape;

    double scale = minVar;
    if(unbiased)
    {
      int n = sample.size();
      shape *= 1.0 - 2.0 / n;
      scale *= 1.0 - invShape / (n - 1);
    }
    SetScale(scale);
    SetShape(shape);
  }

  /**
   * @fn FitShapeBayes
   * Fit α, using bayesian inference
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution of α
   */
  randlib::GammaRand<RealType> FitShapeBayes(const std::vector<RealType>& sample, const randlib::GammaDistribution<RealType>& priorDistribution, bool MAP = false)
  {
    if(!this->allElementsAreNotSmallerThan(sigma, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(sigma)));
    int n = sample.size();
    double newShape = priorDistribution.GetShape() + n;
    double newRate = priorDistribution.GetRate() + n * (this->GetSampleLogMean(sample) - logSigma);
    GammaRand<RealType> posteriorDistribution(newShape, newRate);
    SetShape(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

private:
  static RealType variateForAlphaEqualOne(RandGenerator& randGenerator)
  {
    return 1.0 / randlib::UniformRand<RealType>::StandardVariate(randGenerator);
  }

  static RealType variateForAlphaEqualTwo(RandGenerator& randGenerator)
  {
    return 1.0 / std::sqrt(randlib::UniformRand<RealType>::StandardVariate(randGenerator));
  }

  static RealType variateForGeneralAlpha(double shape, RandGenerator& randGenerator)
  {
    return std::exp(randlib::ExponentialRand<RealType>::StandardVariate(randGenerator) / shape);
  }

  RealType quantileImpl(double p) const override
  {
    double y = logSigma - std::log1pl(-p) / alpha;
    return std::exp(y);
  }

  RealType quantileImpl1m(double p) const override
  {
    double y = logSigma - std::log(p) / alpha;
    return std::exp(y);
  }
};
} // namespace randlib
