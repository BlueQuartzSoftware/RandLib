#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The ZipfRand class <BR>
 * Zipf distribution
 *
 * Notation: X ~ Zipf(s, n)
 */
template <typename IntType = int>
class RANDLIB_EXPORT ZipfRand : public randlib::DiscreteDistribution<IntType>
{
  double s = 2;                 ///< exponent
  IntType n = 1;                ///< number
  double invHarmonicNumber = 1; /// 1 / H(s, n)

  static constexpr int tableSize = 16;
  int hashedVarNum = 1;
  double table[tableSize];

public:
  ZipfRand(double exponent = 2.0, IntType number = 1)
  {
    SetParameters(exponent, number);
  }

  String Name() const override
  {
    return "Zipf(" + this->toStringWithPrecision(GetExponent()) + ", " + this->toStringWithPrecision(GetNumber()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  IntType MinValue() const override
  {
    return 1;
  }

  IntType MaxValue() const override
  {
    return n;
  }

  void SetParameters(double exponent, IntType number)
  {
    if(exponent <= 1.0)
      throw std::invalid_argument("Zipf distribution: exponent should be greater "
                                  "than 1, but it's equal to " +
                                  std::to_string(exponent));
    if(number <= 0)
      throw std::invalid_argument("Zipf distribution: number should be positive, but it's equal to " + std::to_string(number));
    s = exponent;
    n = number;

    invHarmonicNumber = 1.0 / RandMath::harmonicNumber(s, n);

    // WARNING: we calculate pow here and in invHarmonic number
    hashedVarNum = n > tableSize ? tableSize : n;
    table[0] = 1.0;
    for(int i = 1; i < hashedVarNum - 1; ++i)
      table[i] = table[i - 1] + std::pow(i + 1, -s);
    if(hashedVarNum == n)
      table[hashedVarNum - 1] = 1.0 / invHarmonicNumber;
    else
      table[hashedVarNum - 1] = table[hashedVarNum - 2] + std::pow(hashedVarNum, -s);
    for(int i = 0; i < hashedVarNum; ++i)
      table[i] *= invHarmonicNumber;
  }

  inline double GetExponent() const
  {
    return s;
  }

  inline IntType GetNumber() const
  {
    return n;
  }

  double P(const IntType& k) const override
  {
    return (k < 1 || k > n) ? 0.0 : std::pow(k, -s) * invHarmonicNumber;
  }

  double logP(const IntType& k) const override
  {
    return (k < 1 || k > n) ? -INFINITY : -s * std::log(k) + std::log(invHarmonicNumber); // can be hashed
  }

  double F(const IntType& k) const override
  {
    if(k < 1.0)
      return 0.0;
    if(k >= n)
      return 1.0;
    return RandMath::harmonicNumber(s, k) * invHarmonicNumber;
  }

  IntType Variate() const override
  {
    double U = UniformRand<double>::StandardVariate(this->localRandGenerator);
    int k = 1;
    /// if we didn't manage to hash values for such U
    if(U > table[hashedVarNum - 1])
    {
      k = hashedVarNum;
      double sum = table[hashedVarNum - 1];
      do
      {
        ++k;
        sum += std::pow(k, -s) * invHarmonicNumber;
      } while(sum < U);
    }
    else
    {
      while(k <= hashedVarNum && table[k - 1] < U)
        ++k;
    }
    return k;
  }

  long double Mean() const override
  {
    return RandMath::harmonicNumber(s - 1, n) * invHarmonicNumber;
  }

  long double Variance() const override
  {
    double numerator = RandMath::harmonicNumber(s - 1, n);
    numerator *= numerator;
    numerator = RandMath::harmonicNumber(s - 2, n) * RandMath::harmonicNumber(s, n) - numerator;
    return numerator * invHarmonicNumber * invHarmonicNumber;
  }

  IntType Mode() const override
  {
    return 1;
  }

  long double Skewness() const override
  {
    long double harmonic0 = 1.0 / invHarmonicNumber;
    long double harmonic1 = RandMath::harmonicNumber(s - 1, n);
    long double harmonic2 = RandMath::harmonicNumber(s - 2, n);
    long double harmonic3 = RandMath::harmonicNumber(s - 3, n);
    long double first = harmonic3 * harmonic0 * harmonic0;
    long double harmonic2prod0 = harmonic2 * harmonic0;
    long double second = -3 * harmonic1 * harmonic2prod0;
    long double harmonic1Sq = harmonic1 * harmonic1;
    long double third = 2 * harmonic1 * harmonic1Sq;
    long double numerator = first + second + third;
    long double denominator = harmonic2prod0 - harmonic1Sq;
    denominator *= std::sqrt(denominator);
    return numerator / denominator;
  }

  long double ExcessKurtosis() const override
  {
    long double harmonic0 = 1.0 / invHarmonicNumber;
    long double harmonic1 = RandMath::harmonicNumber(s - 1, n);
    long double harmonic2 = RandMath::harmonicNumber(s - 2, n);
    long double harmonic3 = RandMath::harmonicNumber(s - 3, n);
    long double harmonic4 = RandMath::harmonicNumber(s - 4, n);
    long double harmonic2prod0 = harmonic2 * harmonic0;
    long double harmonic0Sq = harmonic0 * harmonic0;
    long double harmonic1Sq = harmonic1 * harmonic1;
    long double denominator = harmonic2prod0 - harmonic1Sq;
    denominator *= denominator;
    long double numerator = harmonic0Sq * harmonic0 * harmonic4;
    numerator -= 4 * harmonic0Sq * harmonic1 * harmonic3;
    numerator += 6 * harmonic2prod0 * harmonic1Sq;
    numerator -= 3 * harmonic1Sq * harmonic1Sq;
    return numerator / denominator - 3;
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> sum(0.0, 0.0);
    for(int i = 1; i <= n; ++i)
    {
      std::complex<double> addon(-s * std::log(i), i * t);
      sum += std::exp(addon);
    }
    return invHarmonicNumber * sum;
  }
};
} // namespace randlib
