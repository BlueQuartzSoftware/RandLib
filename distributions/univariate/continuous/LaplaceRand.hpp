#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

//#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/ExponentialRand.hpp"
//#include "distributions/univariate/continuous/GeometricStableRand.hpp"
#include "distributions/univariate/discrete/BernoulliRand.hpp"

namespace randlib
{
/**
 * @brief The AsymmetricLaplaceDistribution class <BR>
 * Abstract parent class for general Laplace and Asymmetric Laplace
 * distributions
 */
template <typename RealType = double>
class RANDLIB_EXPORT AsymmetricLaplaceDistribution : public GeneralGeometricStableDistribution<RealType>
{
public:
  AsymmetricLaplaceDistribution(double shift = 0, double scale = 1, double asymmetry = 1)
  : GeneralGeometricStableDistribution<RealType>(2.0, 0.0, scale, 0.0, shift)
  {
    GeneralGeometricStableDistribution<RealType>::SetAsymmetry(asymmetry);
    ChangeLocation();
  }

  void SetScale(double scale)
  {
    if(scale <= 0.0)
      throw std::invalid_argument("Laplace distribution: scale should be positive, but it's equal to " + std::to_string(scale));
    GeneralGeometricStableDistribution<RealType>::SetScale(scale);
    ChangeLocation();
  }

  inline double GetAsymmetry() const
  {
    return this->kappa;
  }

  double f(const RealType& x) const override
  {
    return this->pdfLaplace(x - this->m);
  }

  double logf(const RealType& x) const override
  {
    return this->logpdfLaplace(x - this->m);
  }

  double F(const RealType& x) const override
  {
    return this->cdfLaplace(x - this->m);
  }

  double S(const RealType& x) const override
  {
    return this->cdfLaplaceCompl(x - this->m);
  }

  RealType Variate() const override
  {
    RealType X = (this->kappa == 1) ? randlib::LaplaceRand<RealType>::StandardVariate(this->localRandGenerator) :
                                      randlib::AsymmetricLaplaceRand<RealType>::StandardVariate(this->kappa, this->localRandGenerator);
    return this->m + this->gamma * X;
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    if(this->kappa == 1)
    {
      for(RealType& var : outputData)
        var = this->m + this->gamma * randlib::LaplaceRand<RealType>::StandardVariate(this->localRandGenerator);
    }
    else
    {
      for(RealType& var : outputData)
        var = this->m + this->gamma * randlib::AsymmetricLaplaceRand<RealType>::StandardVariate(this->kappa, this->localRandGenerator);
    }
  }

  long double Entropy() const
  {
    double y = this->kappaInv + this->kappa;
    return std::log1pl(this->gamma * y);
  }

  void FitScale(const std::vector<RealType>& sample)
  {
    double deviation = 0.0;
    for(const RealType& x : sample)
    {
      if(x > this->m)
        deviation += this->kappa * (x - this->m);
      else
        deviation -= (x - this->m) / this->kappa;
    }
    deviation /= sample.size();
    SetScale(deviation);
  }

protected:
  void ChangeLocation()
  {
    this->SetLocation((1.0 - this->kappaSq) * this->gamma * this->kappaInv);
  }

private:
  RealType quantileImpl(double p) const override
  {
    return this->quantileLaplace(p);
  }

  RealType quantileImpl1m(double p) const override
  {
    return this->quantileLaplace1m(p);
  }

  std::complex<double> CFImpl(double t) const override
  {
    double bt = this->gamma * t;
    double btSq = bt * bt;
    double denominator = (1 + this->kappaSq * btSq) * (1 + btSq / this->kappaSq);
    std::complex<double> y(std::cos(this->m * t), std::sin(this->m * t));
    std::complex<double> x(1, -this->kappa * bt), z(1, bt * this->kappaInv);
    return x * y * z / denominator;
  }
};

/**
 * @brief The CenteredAsymmetricLaplaceRand class <BR>
 * Centered Laplace distribution
 *
 * Belongs to exponential family
 */
template <typename RealType = double>
class RANDLIB_EXPORT CenteredAsymmetricLaplaceRand : public AsymmetricLaplaceDistribution<RealType>, public ExponentialFamily<RealType, DoublePair>
{
public:
  CenteredAsymmetricLaplaceRand(double scale = 1, double asymmetry = 1)
  : AsymmetricLaplaceDistribution<RealType>(0.0, scale, asymmetry)
  {
  }
};

/**
 * @brief The ShiftedAsymmetricLaplaceDistribution class <BR>
 * Abstract parent class for shifted Laplace and Asymmetric Laplace
 * distributions
 */
template <typename RealType = double>
class RANDLIB_EXPORT ShiftedAsymmetricLaplaceDistribution : public AsymmetricLaplaceDistribution<RealType>
{
public:
  ShiftedAsymmetricLaplaceDistribution(double shift = 0, double scale = 1, double asymmetry = 1)
  : AsymmetricLaplaceDistribution<RealType>(shift, scale, asymmetry)
  {
  }

  using GeneralGeometricStableDistribution<RealType>::SetShift;

  inline double GetShift() const
  {
    return this->m;
  }

  void FitShift(const std::vector<RealType>& sample)
  {
    /// Calculate median (considering asymmetry)
    /// we use root-finding algorithm for median search
    double minVar = *std::min_element(sample.begin(), sample.end());
    double maxVar = *std::max_element(sample.begin(), sample.end());
    double median = this->GetSampleMean(sample);

    if(!RandMath::findRootBrentFirstOrder<double>(
           [this, sample](double med) {
             double y = 0.0;
             for(const RealType& x : sample)
             {
               if(x > med)
                 y -= this->kappaSq;
               else if(x < med)
                 ++y;
             }
             return y;
           },
           minVar, maxVar, median))
      throw std::runtime_error(this->fitErrorDescription(this->UNDEFINED_ERROR, "Error in root-finding procedure"));

    this->SetShift(median);
  }

protected:
  void FitShiftAndScale(const std::vector<RealType>& sample)
  {
    this->FitShift(sample);
    this->FitScale(sample);
  }
};

/**
 * @brief The AsymmetricLaplaceRand class <BR>
 * Asymmetric Laplace distribution
 *
 * Notation: X ~ Asymmetric-Laplace(m, γ, κ)
 *
 * Related distributions: <BR>
 * X = m + γ * (Y / κ - W * κ), where Y, W ~ Exp(1) <BR>
 * X - m ~ GS(2, β, γ, γ(1 - κ^2) / κ) with arbitrary β
 */
template <typename RealType = double>
class RANDLIB_EXPORT AsymmetricLaplaceRand : public ShiftedAsymmetricLaplaceDistribution<RealType>
{
public:
  AsymmetricLaplaceRand(double shift = 0, double scale = 1, double asymmetry = 1)
  : ShiftedAsymmetricLaplaceDistribution<RealType>(shift, scale, asymmetry)
  {
  }

  String Name() const override
  {
    return "Asymmetric-Laplace(" + this->toStringWithPrecision(this->GetShift()) + ", " + this->toStringWithPrecision(this->GetScale()) + ", " + this->toStringWithPrecision(this->GetAsymmetry()) +
           ")";
  }

  void SetAsymmetry(double asymmetry)
  {
    GeneralGeometricStableDistribution<RealType>::SetAsymmetry(asymmetry);
    this->ChangeLocation();
  }

  static RealType StandardVariate(double asymmetry, RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    RealType x = ExponentialRand<RealType>::StandardVariate(randGenerator) / asymmetry;
    RealType y = ExponentialRand<RealType>::StandardVariate(randGenerator) * asymmetry;
    return x - y;
  }

  using ShiftedAsymmetricLaplaceDistribution<RealType>::FitShiftAndScale;

  void FitAsymmetry(const std::vector<RealType>& sample)
  {
    auto [xPlus, xMinus] = getOneSidedSums(sample);
    double gammaN = this->gamma * sample.size();
    double asymmetry = 0;
    if(xPlus == xMinus)
      asymmetry = 1.0;
    else if(xPlus == 0)
      asymmetry = -xMinus / gammaN;
    else if(xMinus == 0)
    {
      asymmetry = gammaN / xPlus;
    }
    else
    {
      /// write down coefficients of quartic equation
      double a = xPlus / gammaN;
      double c = (xPlus + xMinus) / gammaN;
      double e = xMinus / gammaN;
      /// find coefficients for solution
      double ae = a * e;
      double delta1 = 2 * std::pow(c, 3) + 9 * c + 27 * (a + e) - 72 * ae * c;
      double delta0 = c * c + 3 + 12 * ae;
      double p2 = delta1 + std::sqrt(delta1 * delta1 - 4 * std::pow(delta0, 3));
      double Q = std::cbrt(0.5 * p2);
      double temp = 8 * a * a;
      double p = (8 * a * c - 3.0) / temp;
      double q = (-1.0 + 4 * a * c + temp) / (a * temp);
      double S = 0.5 * std::sqrt(-2.0 / 3 * p + (Q + delta0 / Q) / (3 * a));
      /// solve by general formula
      double c1 = -0.25 / a, b1 = -4 * S * S - 2 * p, b2 = q / S;
      if(b1 + b2 > 0)
        asymmetry = c1 + S + 0.5 * std::sqrt(b1 + b2);
      else
        asymmetry = c1 - S + 0.5 * std::sqrt(b1 - b2);
    }

    SetAsymmetry(asymmetry);
  }

private:
  DoublePair getOneSidedSums(const std::vector<RealType>& sample)
  {
    double xPlus = 0.0, xMinus = 0.0;
    for(const RealType& x : sample)
    {
      if(x < this->m)
        xMinus += (x - this->m);
      else
        xPlus += (x - this->m);
    }
    return DoublePair(xPlus, xMinus);
  }
};

/**
 * @brief The LaplaceRand class <BR>
 * Laplace distribution
 *
 * Notation: X ~ Laplace(m, γ)
 *
 * Related distributions: <BR>
 * X = m + γ * (Y - W), where Y, W ~ Exp(1) <BR>
 * X - m ~ GS(2, β, γ, 0) with arbitrary β
 */
template <typename RealType = double>
class RANDLIB_EXPORT LaplaceRand : public ShiftedAsymmetricLaplaceDistribution<RealType>
{
public:
  LaplaceRand(double shift = 0, double scale = 1)
  : ShiftedAsymmetricLaplaceDistribution<RealType>(shift, scale, 1.0)
  {
  }

  String Name() const override
  {
    return "Laplace(" + this->toStringWithPrecision(this->GetShift()) + ", " + this->toStringWithPrecision(this->GetScale()) + ")";
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
    RealType W = ExponentialRand<RealType>::StandardVariate(randGenerator);
    return BernoulliRand::StandardVariate(randGenerator) ? W : -W;
  }

  void Fit(const std::vector<RealType>& sample)
  {
    this->FitShiftAndScale(sample);
  }
};
} // namespace randlib
