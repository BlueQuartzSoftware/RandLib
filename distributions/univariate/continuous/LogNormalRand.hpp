#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/NormalRand.hpp"

namespace randlib
{
/**
 * @brief The LogNormalRand class <BR>
 * Log-Normal distribution
 *
 * Notation X ~ Log-Normal(μ, σ)
 *
 * Related distributions: <BR>
 * ln(X) ~ Normal(μ, σ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT LogNormalRand : public randlib::ContinuousDistribution<RealType>
{
  NormalRand<RealType> X{};
  double expMu = 1;                     ///< exp(μ)
  double expHalfSigmaSq = 1.6487212707; ///< exp(σ^2 / 2)

public:
  LogNormalRand(double location = 0, double squaredScale = 1)
  {
      SetLocation(location);
      SetScale(squaredScale > 0.0 ? std::sqrt(squaredScale) : 1.0);
  }

  String Name() const override
  {
      return "Log-Normal(" + this->toStringWithPrecision(GetLocation()) + ", " + this->toStringWithPrecision(GetScale()) + ")";
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

  void SetLocation(double location)
  {
      X.SetLocation(location);
      expMu = std::exp(X.GetLocation());
  }

  void SetScale(double scale)
  {
      if(scale <= 0.0)
          throw std::invalid_argument("Log-Normal distribution: scale should be positive");
      X.SetScale(scale);
      expHalfSigmaSq = std::exp(0.5 * X.Variance());
  }

  inline double GetLocation() const
  {
    return X.Mean();
  }

  inline double GetScale() const
  {
    return X.GetScale();
  }

  double f(const RealType& x) const override
  {
      return (x > 0.0) ? std::exp(logf(x)) : 0.0;
  }

  double logf(const RealType& x) const override
  {
      if(x <= 0.0)
          return -INFINITY;
      double logX = std::log(x);
      double y = X.logf(logX);
      return y - logX;
  }

  double F(const RealType& x) const override
  {
      return (x > 0.0) ? X.F(std::log(x)) : 0.0;
  }

  double S(const RealType& x) const override
  {
      return (x > 0.0) ? X.S(std::log(x)) : 1.0;
  }

  RealType Variate() const override
  {
      return std::exp(X.Variate());
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      return std::exp(NormalRand<RealType>::StandardVariate(randGenerator));
  }

  void Reseed(unsigned long seed) const override
  {
      X.Reseed(seed);
  }

  long double Mean() const override
  {
      return expMu * expHalfSigmaSq;
  }

  long double Variance() const override
  {
      double y = expMu * expHalfSigmaSq;
      return y * y * std::expm1l(X.Variance());
  }

    RealType Median() const override
    {
        return expMu;
    }

  RealType Mode() const override
  {
      return expMu / (expHalfSigmaSq * expHalfSigmaSq);
  }

  long double Skewness() const override
  {
      double y = std::expm1l(X.Variance());
      return (expHalfSigmaSq * expHalfSigmaSq + 2) * std::sqrt(y);
  }

  long double ExcessKurtosis() const override
  {
      double temp = expHalfSigmaSq * expHalfSigmaSq;
      double c = temp * temp;
      double b = c * temp;
      double a = c * c;
      return a + 2 * b + 3 * c - 6;
  }

    /**
    * @fn FitLocation
    * Fit location via maximum-likelihood
    * @param sample
    */
    void FitLocation(const std::vector<RealType>& sample)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        this->SetLocation(this->GetSampleLogMean(sample));
    }

    /**
     * @fn FitScale
     * Fit scale via maximum-likelihood
     * @param sample
     */
    void FitScale(const std::vector<RealType>& sample)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        double mu = X.GetLocation();
        this->SetScale(std::sqrt(this->GetSampleLogVariance(sample, mu)));
    }

    /**
     * @fn Fit
     * Fit parameters via maximum-likelihood
     * @param sample
     */
    void Fit(const std::vector<RealType>& sample)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        size_t n = sample.size();
        long double logMean = 0.0L;
        long double logSqDev = 0.0L;
        for(double var : sample)
        {
            double logVar = std::log(var);
            logMean += logVar;
            logSqDev += logVar * logVar;
        }
        logMean /= n;
        logSqDev /= n;
        logSqDev -= logMean * logMean;

        SetLocation(logMean);
        SetScale(std::sqrt(logSqDev));
    }

    /**
     * @fn FitLocationBayes
     * Set location, using bayesian inference
     * @param sample
     * @param priorDistribution
     * @param MAP if true, use MAP estimator
     * @return posterior distribution
     */
    NormalRand<RealType> FitLocationBayes(const std::vector<RealType>& sample, const NormalRand<RealType>& priorDistribution, bool MAP = false)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        size_t n = sample.size();
        double mu0 = priorDistribution.GetLocation();
        double tau0 = priorDistribution.GetPrecision();
        double tau = X.GetPrecision();
        double numerator = n * this->GetSampleLogMean(sample) * tau + tau0 * mu0;
        double denominator = n * tau + tau0;
        NormalRand<RealType> posteriorDistribution(numerator / denominator, 1.0 / denominator);
        SetLocation(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
        return posteriorDistribution;
    }

    /**
     * @fn FitScaleBayes
     * Set scale, using bayesian inference
     * @param sample
     * @param priorDistribution
     * @param MAP if true, use MAP estimator
     * @return posterior distribution
     */
    InverseGammaRand<RealType> FitScaleBayes(const std::vector<RealType>& sample, const InverseGammaRand<RealType>& priorDistribution, bool MAP = false)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        size_t n = sample.size();
        double alpha = priorDistribution.GetShape();
        double beta = priorDistribution.GetRate();
        double newAlpha = alpha + 0.5 * n;
        double mu = X.GetLocation();
        double newBeta = beta + 0.5 * n * this->GetSampleLogVariance(sample, mu);
        InverseGammaRand<RealType> posteriorDistribution(newAlpha, newBeta);
        SetScale(std::sqrt(MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean()));
        return posteriorDistribution;
    }

    /**
     * @fn FitBayes
     * Set parameters, using bayesian inference
     * @param sample
     * @param priorDistribution
     * @param MAP if true, use MAP estimator
     * @return posterior distribution
     */
    NormalInverseGammaRand<RealType> FitBayes(const std::vector<RealType>& sample, const NormalInverseGammaRand<RealType>& priorDistribution, bool MAP = false)
    {
        /// Sanity check
        if(!this->allElementsArePositive(sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->POSITIVITY_VIOLATION));
        size_t n = sample.size();
        double alpha = priorDistribution.GetShape();
        double beta = priorDistribution.GetRate();
        double mu0 = priorDistribution.GetLocation();
        double lambda = priorDistribution.GetPrecision();
        DoublePair logStats = this->GetSampleLogMeanAndVariance(sample);
        double average = logStats.first, sum = n * average;
        double newLambda = lambda + n;
        double newMu0 = (lambda * mu0 + sum) / newLambda;
        double newAlpha = alpha + 0.5 * n;
        double variance = logStats.second;
        double aux = mu0 - average;
        double newBeta = beta + 0.5 * n * (variance + lambda / newLambda * aux * aux);
        NormalInverseGammaRand<RealType> posteriorDistribution(newMu0, newLambda, newAlpha, newBeta);
        DoublePair newParams = MAP ? static_cast<DoublePair>(posteriorDistribution.Mode()) : static_cast<DoublePair>(posteriorDistribution.Mean());
        SetLocation(newParams.first);
        SetScale(std::sqrt(newParams.second));
        return posteriorDistribution;
    }

private:
  RealType quantileImpl(double p) const override
  {
      return std::exp(X.Quantile(p));
  }

  RealType quantileImpl1m(double p) const override
  {
      return std::exp(X.Quantile1m(p));
  }
};
} // namespace randlib
