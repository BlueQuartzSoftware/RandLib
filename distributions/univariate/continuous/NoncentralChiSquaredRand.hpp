#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/discrete/PoissonRand.hpp"

namespace randlib
{
/**
 * @brief The NoncentralChiSquaredRand class <BR>
 * Noncentral Chi-Squared distribution
 *
 * Notation: X ~ χ'^2(k, λ)
 *
 * Related distributions: <BR>
 * If X ~ χ'^2(k, 0), then X ~ χ^2(k) <BR>
 * X ~ χ^2(k + 2J), where J ~ Po(λ)
 */
template <typename RealType = double>
class RANDLIB_EXPORT NoncentralChiSquaredRand : public randlib::ContinuousDistribution<RealType>
{
  double k = 1;                ///< degree
  double lambda = 2;           ///< noncentrality λ
  double halfK = 0.5;          ///< k / 2
  double halfLambda = 1;       ///< λ / 2
  double sqrtLambda = M_SQRT2; ///< √λ
  double logLambda = M_LN2;    ///< log(λ)

  PoissonRand<int> Y{};

public:
  explicit NoncentralChiSquaredRand(double degree = 1, double noncentrality = 0)
  {
    SetParameters(degree, noncentrality);
  }

  String Name() const override
  {
    return "Noncentral Chi-Squared(" + this->toStringWithPrecision(GetDegree()) + ", " + this->toStringWithPrecision(GetNoncentrality()) + ")";
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

  void SetParameters(double degree, double noncentrality)
  {
    if(degree <= 0.0)
      throw std::invalid_argument("Noncentral Chi-Squared distribution: degree "
                                  "parameter should be positive");
    if(noncentrality <= 0.0)
      throw std::invalid_argument("Noncentral Chi-Squared distribution: "
                                  "noncentrality parameter should be positive");

    k = degree;
    halfK = 0.5 * k;

    lambda = noncentrality;
    halfLambda = 0.5 * lambda;
    sqrtLambda = std::sqrt(lambda);
    logLambda = std::log(lambda);

    if(k < 1)
      Y.SetRate(halfLambda);
  }

  inline double GetDegree() const
  {
    return k;
  }

  inline double GetNoncentrality() const
  {
    return lambda;
  }

  double f(const RealType& x) const override
  {
    if(x < 0.0)
      return 0.0;
    if(x == 0.0)
    {
      if(k == 2)
        return 0.5 * std::exp(-halfLambda);
      return (k > 2) ? 0.0 : INFINITY;
    }
    return std::exp(logf(x));
  }

  double logf(const RealType& x) const override
  {
    if(x < 0.0)
      return -INFINITY;
    if(x == 0.0)
    {
      if(k == 2)
        return -halfLambda - M_LN2;
      return (k > 2) ? -INFINITY : INFINITY;
    }
    double halfKm1 = halfK - 1;
    double y = RandMath::logBesselI(halfKm1, std::sqrt(lambda * x));
    double z = halfKm1 * (std::log(x) - logLambda);
    z -= x + lambda;
    return y + 0.5 * z - M_LN2;
  }

  double F(const RealType& x) const override
  {
    if(x <= 0.0)
      return 0.0;
    double halfX = 0.5 * x;
    double sqrtHalfX = std::sqrt(halfX);
    double logHalfX = std::log(halfX);
    return RandMath::MarcumP(halfK, halfLambda, halfX, sqrtLambda / M_SQRT2, sqrtHalfX, logLambda - M_LN2, logHalfX);
  }

  double S(const RealType& x) const override
  {
    if(x <= 0.0)
      return 1.0;
    double halfX = 0.5 * x;
    double sqrtHalfX = std::sqrt(halfX);
    double logHalfX = std::log(halfX);
    return RandMath::MarcumQ(halfK, halfLambda, halfX, sqrtLambda / M_SQRT2, sqrtHalfX, logLambda - M_LN2, logHalfX);
  }

  static RealType Variate(double degree, double noncentrality, RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    if(degree <= 0.0)
      throw std::invalid_argument("Noncentral Chi-Squared distribution: degree "
                                  "parameter should be positive");
    if(noncentrality <= 0.0)
      throw std::invalid_argument("Noncentral Chi-Squared distribution: "
                                  "noncentrality parameter should be positive");

    if(degree >= 1)
    {
      RealType rv = (degree == 1) ? 0.0 : 2 * GammaDistribution<RealType>::StandardVariate(0.5 * degree - 0.5, randGenerator);
      RealType y = std::sqrt(noncentrality) + NormalRand<RealType>::StandardVariate(randGenerator);
      return rv + y * y;
    }
    RealType shape = 0.5 * degree + PoissonRand<int>::Variate(0.5 * noncentrality, randGenerator);
    return 2 * GammaDistribution<RealType>::StandardVariate(shape, randGenerator);
  }

  RealType Variate() const override
  {
    if(k < 1)
      return 2 * GammaDistribution<RealType>::StandardVariate(halfK + Y.Variate(), this->localRandGenerator);
    double X = variateForDegreeEqualOne();
    if(k > 1)
      X += 2 * GammaDistribution<RealType>::StandardVariate(halfK - 0.5, this->localRandGenerator);
    return X;
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    if(k >= 1)
    {
      for(RealType& var : outputData)
        var = variateForDegreeEqualOne();
      double halfKmHalf = halfK - 0.5;
      if(halfKmHalf == 0)
        return;
      for(RealType& var : outputData)
        var += 2 * GammaDistribution<RealType>::StandardVariate(halfKmHalf, this->localRandGenerator);
    }
    else
    {
      for(RealType& var : outputData)
        var = 2 * GammaDistribution<RealType>::StandardVariate(halfK + Y.Variate(), this->localRandGenerator);
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    Y.Reseed(seed + 1);
  }

  long double Mean() const override
  {
    return k + lambda;
  }

  long double Variance() const override
  {
    return 2 * (k + 2 * lambda);
  }

  RealType Mode() const override
  {
    return (k <= 2) ? 0.0 : randlib::ContinuousDistribution<RealType>::Mode();
  }

  long double Skewness() const override
  {
    long double y = k + 2 * lambda;
    y = 2.0 / y;
    long double z = y * std::sqrt(y);
    return z * (k + 3 * lambda);
  }

  long double ExcessKurtosis() const override
  {
    long double y = k + 2 * lambda;
    return 12 * (k + 4 * lambda) / (y * y);
  }

private:
  RealType variateForDegreeEqualOne() const
  {
    RealType y = sqrtLambda + NormalRand<RealType>::StandardVariate(this->localRandGenerator);
    return y * y;
  }

  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> aux(1, -2 * t);
    std::complex<double> y(0, lambda * t);
    y /= aux;
    y -= halfK * std::log(aux);
    return std::exp(y);
  }
};
} // namespace randlib
