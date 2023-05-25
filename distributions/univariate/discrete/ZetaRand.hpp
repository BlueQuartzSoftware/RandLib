#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/DiscreteDistributions.hpp"

namespace randlib
{
/**
 * @brief The ZetaRand class <BR>
 * Zeta distribution
 *
 * P(X = k) = 1 / (k^s * ζ(s))
 *
 * Notation: X ~ Zeta(s)
 */
template <typename IntType = int>
class RANDLIB_EXPORT ZetaRand : public randlib::DiscreteDistribution<IntType>
{
  double s = 2;                                 ///< exponent
  double sm1 = 1;                               ///< s - 1
  double zetaS = M_PI_SQ / 6.0;                 ///< ζ(s), where ζ stands for Riemann zeta-function
  double logZetaS = 2 * M_LNPI - M_LN2 - M_LN3; ///< ln(ζ(s))
  double b = 0.5;                               ///< 1 - 2^(1-s)

public:
  explicit ZetaRand(double exponent = 2.0)
  {
      SetExponent(exponent);
  }

  String Name() const override
  {
      return "Zeta(" + this->toStringWithPrecision(GetExponent()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }

  IntType MinValue() const override
  {
    return 1;
  }

  IntType MaxValue() const override
  {
    return std::numeric_limits<IntType>::max();
  }

  void SetExponent(double exponent)
  {
      if(exponent <= 1.0)
          throw std::invalid_argument("Zeta distribution: exponent should be greater "
                                      "than 1, but it's equal to " +
                                      std::to_string(exponent));
      s = exponent;
      sm1 = s - 1.0;
      zetaS = std::riemann_zetal(s);
      logZetaS = std::log(zetaS);
      b = -std::expm1l(-sm1 * M_LN2);
  }

  inline double GetExponent() const
  {
    return s;
  }

  double logP(const IntType& k) const override
  {
      return (k < 1) ? -INFINITY : -logZetaS - s * std::log(k);
  }

  double F(const IntType& k) const override
  {
      return (k < 1) ? 0.0 : RandMath::harmonicNumber(s, k) / zetaS;
  }

  IntType Variate() const override
  {
      /// Luc Devroye, p. 551
      /// rejection sampling from rounded down Pareto distribution
      size_t iter = 0;
      do
      {
          float X = std::floor(ParetoRand<float>::StandardVariate(sm1, this->localRandGenerator));
          float V = UniformRand<float>::StandardVariate(this->localRandGenerator);
          if(X < 1e4)
          {
              float T = std::pow(1.0 + 1.0 / X, sm1);
              /// there was a typo in the book - '<=' instead of '>'
              if(V * X * (T - 1) <= b * T)
                  return X;
          }
          else
          {
              long double logT = sm1 * std::log1pl(1.0 / X);
              long double TM1 = std::expm1l(logT);
              if(V * X * TM1 <= b * std::exp(logT))
                  return X < this->MaxValue() ? (IntType)X : this->MaxValue();
          }

      } while(++iter <= ProbabilityDistribution<IntType>::MAX_ITER_REJECTION);
      throw std::runtime_error("Zeta distribution: sampling failed");
  }

  long double Mean() const override
  {
      return (s > 2) ? std::riemann_zetal(sm1) / zetaS : INFINITY;
  }

  long double Variance() const override
  {
      if(s <= 3)
          return INFINITY;
      double y = Mean();
      double z = std::riemann_zetal(s - 2) / zetaS;
      return z - y * y;
  }

  IntType Mode() const override
  {
      return 1;
  }

  long double Skewness() const override
  {
      if(s <= 4)
          return INFINITY;
      long double z1 = std::riemann_zetal(sm1), z1Sq = z1 * z1;
      long double z2 = std::riemann_zetal(s - 2);
      long double z3 = std::riemann_zetal(s - 3);
      long double z = zetaS, zSq = z * z;
      long double logskew = zSq * z3;
      logskew -= 3 * z2 * z1 * z;
      logskew += 2 * z1 * z1Sq;
      logskew = std::log(logskew);
      logskew -= 1.5 * std::log(z * z2 - z1Sq);
      logskew -= 2 * logZetaS;
      return std::exp(logskew);
  }

  long double ExcessKurtosis() const override
  {
      if(s <= 5)
          return INFINITY;
      long double mean = Mean();
      long double secondMoment = this->SecondMoment();
      long double thirdMoment = ThirdMoment();
      long double fourthMoment = FourthMoment();
      long double meanSq = mean * mean;
      long double variance = secondMoment - meanSq;
      long double numerator = fourthMoment - 4 * thirdMoment * mean + 6 * secondMoment * meanSq - 3 * meanSq * meanSq;
      long double denominator = variance * variance;
      return numerator / denominator - 3.0;
  }

  long double Moment(int n) const
  {
      return (s > n + 1) ? std::riemann_zetal(s - n) / zetaS : INFINITY;
  }

  long double ThirdMoment() const override
  {
    return Moment(3);
  }

  long double FourthMoment() const override
  {
    return Moment(4);
  }

  inline long double GetZetaFunction() const
  {
    return zetaS;
  }

  inline long double GetLogZetaFunction() const
  {
    return logZetaS;
  }
};
} // namespace randlib
