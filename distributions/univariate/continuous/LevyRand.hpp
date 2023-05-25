#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/StableRand.hpp"

namespace randlib
{
/**
 * @brief The LevyRand class <BR>
 * Levy distribution
 *
 * f(x | μ, σ) = ((σ exp(σ / (μ - x)) / (2 π (x - μ)^3))^(1/2)
 *
 * Notation: X ~ Levy(μ, σ)
 *
 * Related distributions: <BR>
 * If X ~ Levy(0, 1), then μ + σ * X ~ Levy(μ, σ) <BR>
 * X ~ S(0.5, 1, σ, μ) <BR>
 * If Y ~ Normal(0, 1), then 1 / X^2 ~ Levy(0, 1) <BR>
 * If X ~ Levy(0, σ), then X ~ Inv-Γ(1/2, σ/2)
 */
template <typename RealType = double>
class RANDLIB_EXPORT LevyRand : public StableDistribution<RealType>
{
public:
  LevyRand(double location = 0, double scale = 1)
          : StableDistribution<RealType>(0.5, 1, scale, location)
  {
  }

  String Name() const override
  {
      return "Levy(" + this->toStringWithPrecision(this->GetLocation()) + ", " + this->toStringWithPrecision(this->GetScale()) + ")";
  }

    double f(const RealType& x) const override
    {
        return this->pdfLevy(x);
    }

  double logf(const RealType& x) const override
  {
      return this->logpdfLevy(x);
  }

  double F(const RealType& x) const override
  {
      return this->cdfLevy(x);
  }

  double S(const RealType& x) const override
  {
      return this->cdfLevyCompl(x);
  }

  RealType Variate() const override
  {
      RealType rv = NormalRand<RealType>::StandardVariate(this->localRandGenerator);
      rv *= rv;
      rv = this->gamma / rv;
      return this->mu + rv;
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      RealType rv = NormalRand<RealType>::StandardVariate(randGenerator);
      return 1.0 / (rv * rv);
  }

    /**
     * @fn FitScale
     * Fit scale using maximum-likelihoood estimator
     * @param sample
     */
    void FitScale(const std::vector<RealType>& sample)
    {
        /// Sanity check
        if(!this->allElementsAreNotSmallerThan(this->mu, sample))
            throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->mu)));
        long double invSum = 0.0;
        for(RealType var : sample)
            invSum += 1.0 / (var - this->mu);
        invSum = 1.0 / invSum;
        this->SetScale(sample.size() * invSum);
    }

private:
  RealType quantileImpl(double p) const override
  {
      return this->quantileLevy(p);
  }

  RealType quantileImpl1m(double p) const override
  {
      return this->quantileLevy1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
      return this->cfLevy(t);
  }
};
} // namespace randlib
