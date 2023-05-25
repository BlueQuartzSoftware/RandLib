#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/BetaRand.hpp"

namespace randlib
{
/**
 * @brief The BetaPrimeRand class <BR>
 * Beta-prime distribution
 *
 * f(x | α, β) = x^{α-1} (1 + x)^{-α - β} / B(α, β), <BR>
 * where B(α, β) denotes Beta function
 *
 * Notation: X ~ B'(α, β)
 *
 * Related distributions: <BR>
 * X / (X + 1) ~ B(α, β) <BR>
 * X = Y / Z, where Y ~ Γ(α) and Z ~ Γ(β) <BR>
 * β/α * X ~ F(2α, 2β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT BetaPrimeRand : public randlib::ContinuousDistribution<RealType>
{
  double alpha = 1; ///< first shape α
  double beta = 1;  ///< second shape β
  BetaRand<RealType> B{};

public:
  BetaPrimeRand(double shape1 = 1, double shape2 = 1)
  {
    SetShapes(shape1, shape2);
  }

  String Name() const override
  {
    return "Beta Prime(" + this->toStringWithPrecision(GetAlpha()) + ", " + this->toStringWithPrecision(GetBeta()) + ")";
  }

  void SetShapes(double shape1, double shape2)
  {
    if(shape1 <= 0 || shape2 <= 0)
      throw std::invalid_argument("Beta-prime distribution: shapes should be positive");
    B.SetShapes(shape1, shape2);
    alpha = B.GetAlpha();
    beta = B.GetBeta();
  }

  inline double GetAlpha() const
  {
    return alpha;
  }

  inline double GetBeta() const
  {
    return beta;
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

  double f(const RealType& x) const override
  {
    if(x < 0.0)
      return 0.0;
    if(x == 0.0)
    {
      if(alpha == 1.0)
        return 1.0 / GetBetaFunction();
      return (alpha > 1) ? 0.0 : INFINITY;
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0.0)
    {
      if(alpha == 1.0)
        return -GetLogBetaFunction();
      return (alpha > 1) ? -INFINITY : INFINITY;
    }
    double y = (alpha - 1) * std::log(x);
    y -= (alpha + beta) * std::log1pl(x);
    return y - GetLogBetaFunction();
  }

  double F(const RealType& x) const override
  {
    return (x > 0) ? B.F(x / (1.0 + x)) : 0;
  }

  double S(const RealType& x) const override
  {
    return (x > 0) ? B.S(x / (1.0 + x)) : 1;
  }

  RealType Variate() const override
  {
    double x = B.Variate();
    return fromBetaVariate(x);
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    B.Sample(outputData);
    for(RealType& var : outputData)
      var = fromBetaVariate(var);
  }

  void Reseed(unsigned long seed) const override
  {
    B.Reseed(seed);
  }

  long double Mean() const override
  {
    return (beta > 1) ? alpha / (beta - 1) : INFINITY;
  }

  long double Variance() const override
  {
    if(beta <= 2)
      return INFINITY;
    double betam1 = beta - 1;
    double numerator = alpha * (alpha + betam1);
    double denominator = (betam1 - 1) * betam1 * betam1;
    return numerator / denominator;
  }

  RealType Median() const override
  {
    return (alpha == beta) ? 1.0 : quantileImpl(0.5);
  }

  RealType Mode() const override
  {
    return (alpha < 1) ? 0 : (alpha - 1) / (beta + 1);
  }

  long double Skewness() const override
  {
    if(beta <= 3)
      return INFINITY;
    long double aux = alpha + beta - 1;
    long double skewness = (beta - 2) / (alpha * aux);
    skewness = std::sqrt(skewness);
    aux += alpha;
    aux += aux;
    return aux * skewness / (beta - 3);
  }

  long double ExcessKurtosis() const override
  {
    if(beta <= 4)
      return INFINITY;
    long double betam1 = beta - 1;
    long double numerator = betam1 * betam1 * (beta - 2) / (alpha * (alpha + betam1));
    numerator += 5 * beta - 11;
    long double denominator = (beta - 3) * (beta - 4);
    return 6 * numerator / denominator;
  }

  /**
   * @fn GetBetaFunction
   * @return B(α, β)
   */
  inline double GetBetaFunction() const
  {
    return B.GetBetaFunction();
  }

  /**
   * @fn GetLogBetaFunction
   * @return log(B(α, β))
   */
  inline double GetLogBetaFunction() const
  {
    return B.GetLogBetaFunction();
  }

  /**
   * @fn FitAlpha
   * fit α by maximum-likelihood
   * @param sample
   */
  void FitAlpha(const std::vector<RealType>& sample)
  {
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    long double lnG = this->GetSampleLogMean(sample) - B.GetSampleLog1pMeanNorm(sample);
    long double mean = 0.5;
    if(beta != 1.0)
    {
      mean = 0.0;
      for(const double& var : sample)
        mean += var / (1.0 + var);
      mean /= sample.size();
    }
    B.FitAlpha(lnG, mean);
    SetShapes(B.GetAlpha(), beta);
  }

  /**
   * @fn FitBeta
   * fit β by maximum-likelihood
   * @param sample
   */
  void FitBeta(const std::vector<RealType>& sample)
  {
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    long double lnG1m = -B.GetSampleLog1pMeanNorm(sample);
    long double mean = 0.5;
    if(alpha != 1.0)
    {
      mean = 0.0;
      for(const double& var : sample)
        mean += var / (1.0 + var);
      mean /= sample.size();
    }
    B.FitBeta(lnG1m, mean);
    SetShapes(alpha, B.GetBeta());
  }

  /**
   * @fn Fit
   * fit shapes by maximum-likelihood
   * @param sample
   */
  void Fit(const std::vector<RealType>& sample)
  {
    if(!this->allElementsArePositive(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
    long double lnG1m = -B.GetSampleLog1pMeanNorm(sample);
    long double lnG = this->GetSampleLogMean(sample) + lnG1m;
    long double m = 0.0, v = 0.0;
    int n = sample.size();
    for(int i = 0; i < n; ++i)
    {
      double x = sample[i] / (1.0 + sample[i]);
      double diff = x - m;
      m += diff / (i + 1);
      v += diff * (x - m);
    }
    B.FitShapes(lnG, lnG1m, m, v / n);
    SetShapes(B.GetAlpha(), B.GetBeta());
  }

private:
  RealType fromBetaVariate(const RealType& betaVar) const
  {
    if(betaVar > 1e-5)
      return betaVar / (1.0 - betaVar);
    RealType logVar = std::log(betaVar), log1mVar = std::log1p(-betaVar);
    return std::exp(logVar - log1mVar);
  }

  RealType quantileImpl(double p) const override
  {
    double x = B.Quantile(p);
    return x / (1.0 - x);
  }

  RealType quantileImpl1m(double p) const override
  {
    double x = B.Quantile1m(p);
    return x / (1.0 - x);
  }

  std::complex<double> CFImpl(double t) const override
  {
    /// if no singularity - simple numeric integration
    if(alpha >= 1)
      return randlib::UnivariateDistribution<RealType>::CFImpl(t);

    double re = this->ExpectedValue(
                    [this, t](double x) {
                      if(x == 0.0)
                      {
                        return 0.0;
                      }
                      return std::cos(t * x) - 1.0;
                    },
                    0.0, INFINITY) +
                1.0;

    double im = this->ExpectedValue([this, t](double x) { return std::sin(t * x); }, 0.0, INFINITY);
    return std::complex<double>(re, im);
  }
};
} // namespace randlib
