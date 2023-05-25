#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

namespace randlib
{
/**
 * @brief The SkellamRand class <BR>
 * Skellam distribution
 *
 * Notation: X ~ Skellam(μ1, μ2)
 *
 * Related distributions: <BR>
 * If Y ~ Po(μ1) and Z ~ Po(μ2) then Y - Z ~ Skellam(μ1, μ2)
 */
template <typename IntType = int>
class RANDLIB_EXPORT SkellamRand : public randlib::DiscreteDistribution<IntType>
{
  double mu1 = 1;     ///< first rate μ1
  double mu2 = 1;     ///< second rate μ2
  double logMu1 = 0;  ///< log(μ1)
  double logMu2 = 0;  ///< log(μ2)
  double sqrtMu1 = 1; ///< √μ1
  double sqrtMu2 = 1; ///< √μ2

  PoissonRand<IntType> X{}, Y{};

public:
  SkellamRand(double rate1 = 1.0, double rate2 = 1.0)
  {
      SetRates(rate1, rate2);
  }

  String Name() const override
  {
      return "Skellam(" + this->toStringWithPrecision(GetFirstRate()) + ", " + this->toStringWithPrecision(GetSecondRate()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::INFINITE_T;
  }

  IntType MinValue() const override
  {
    return std::numeric_limits<IntType>::lowest();
  }

  IntType MaxValue() const override
  {
    return std::numeric_limits<IntType>::max();
  }

  void SetRates(double rate1, double rate2)
  {
      if(rate1 <= 0.0 || rate2 <= 0.0)
          throw std::invalid_argument("Skellam distribution: rates should be positive");

      X.SetRate(rate1);
      mu1 = X.GetRate();
      logMu1 = std::log(mu1);
      sqrtMu1 = std::sqrt(mu1);

      Y.SetRate(rate2);
      mu2 = Y.GetRate();
      logMu2 = std::log(mu2);
      sqrtMu2 = std::sqrt(mu2);
  }

  inline double GetFirstRate() const
  {
    return mu1;
  }

  inline double GetSecondRate() const
  {
    return mu2;
  }

  double P(const IntType& k) const override
  {
      return std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
      double y = RandMath::logBesselI(k, 2 * sqrtMu1 * sqrtMu2);
      y += 0.5 * k * (logMu1 - logMu2);
      y -= mu1 + mu2;
      return y;
  }

  double F(const IntType& k) const override
  {
      return (k < 0) ? RandMath::MarcumP(-k, mu1, mu2, sqrtMu1, sqrtMu2, logMu1, logMu2) : RandMath::MarcumQ(k + 1, mu2, mu1, sqrtMu2, sqrtMu1, logMu2, logMu1);
  }

  double S(const IntType& k) const override
  {
      return (k < 0) ? RandMath::MarcumQ(-k, mu1, mu2, sqrtMu1, sqrtMu2, logMu1, logMu2) : RandMath::MarcumP(k + 1, mu2, mu1, sqrtMu2, sqrtMu1, logMu2, logMu1);
  }

  IntType Variate() const override
  {
      return X.Variate() - Y.Variate();
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
      X.Sample(outputData);
      for(IntType& var : outputData)
          var -= Y.Variate();
  }

  void Reseed(unsigned long seed) const override
  {
      X.Reseed(seed);
      Y.Reseed(seed + 1);
  }

  long double Mean() const override
  {
      return mu1 - mu2;
  }

  long double Variance() const override
  {
      return mu1 + mu2;
  }

  IntType Median() const override
  {
      return randlib::DiscreteDistribution<IntType>::quantileImpl(0.5, mu1 - mu2);
  }

  IntType Mode() const override
  {
      return Mean();
  }

  long double Skewness() const override
  {
      return (mu1 - mu2) / std::pow(mu1 + mu2, 1.5);
  }

  long double ExcessKurtosis() const override
  {
      return 1.0 / (mu1 + mu2);
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
      double cosT = std::cos(t), sinT = std::sin(t);
      double x = (cosT - 1) * (mu1 + mu2);
      double y = sinT * (mu1 - mu2);
      std::complex<double> z(x, y);
      return std::exp(z);
  }
};
} // namespace randlib
