#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The IrwinHallRand class <BR>
 * Irwin-Hall distribution
 *
 * f(x | n) = 0.5 / (n - 1)! * sum_{k=0}^n (-1)^k * C(n,k) * (x - k) ^ (n - 1) *
 * sign(x - k)
 *
 * Notation: X ~ IH(n)
 *
 * Related distributions: <BR>
 * X ~ Y_1 + Y_2 + ... + Y_n, where Y_i ~ U(0,1)
 */
template <typename RealType = double>
class RANDLIB_EXPORT IrwinHallRand : public randlib::ContinuousDistribution<RealType>
{
  int n = 1; ///< parameter of the distribution
  UniformRand<RealType> U{};

public:
  explicit IrwinHallRand(size_t number)
  {
      SetNumber(number);
  }

    String Name() const override
    {
        return "Irwin-Hall(" + this->toStringWithPrecision(GetNumber()) + ")";
    }

  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return 0;
  }

  RealType MaxValue() const override
  {
    return n;
  }

  void SetNumber(int number)
  {
      if(number <= 0)
          throw std::invalid_argument("Irwin-Hall distribution: number should be positive");
      n = number;
  }

  inline int GetNumber() const
  {
    return n;
  }

  double f(const RealType& x) const override
  {
      if(x < 0 || x > n)
          return 0.0;

      /// Check simplest cases
      if(n == 1)
          return 1.0;
      if(n == 2)
          return (x <= 1.0) ? x : 2.0 - x;
      /// General case
      double sum = 0.0;
      int last = std::floor(x);
      for(int i = 0; i <= last; ++i)
      {
          double add = (n - 1) * std::log(x - i);
          add -= RandMath::lfact(i);
          add -= RandMath::lfact(n - i);
          add = std::exp(add);
          sum += (i % 2) ? -add : add;
      }
      return n * sum;
  }

  double logf(const RealType& x) const override
  {
      return std::log(f(x));
  }

  double F(const RealType& x) const override
  {
      if(x <= 0)
          return 0.0;
      if(x >= n)
          return 1.0;

      /// Check simplest cases
      if(n == 1)
          return x;
      if(n == 2)
      {
          if(x <= 1.0)
              return 0.5 * x * x;
          double temp = 2.0 - x;
          return 1.0 - 0.5 * temp * temp;
      }
      /// General case
      double sum = 0.0;
      int last = std::floor(x);
      for(int i = 0; i <= last; ++i)
      {
          double add = n * std::log(x - i);
          add -= RandMath::lfact(i);
          add -= RandMath::lfact(n - i);
          add = std::exp(add);
          sum += (i % 2) ? -add : add;
      }
      return sum;
  }

  RealType Variate() const override
  {
      RealType sum = 0.0;
      for(int i = 0; i != n; ++i)
          sum += U.Variate();
      return sum;
  }

  void Reseed(unsigned long seed) const override
  {
      U.Reseed(seed);
  }

  long double Mean() const override
  {
      return 0.5 * n;
  }

  long double Variance() const override
  {
      static constexpr long double M_1_12 = 0.08333333333333l;
      return n * M_1_12;
  }

  RealType Median() const override
  {
      return 0.5 * n;
  }

  RealType Mode() const override
  {
      return 0.5 * n;
  }

  long double Skewness() const override
  {
      return 0.0l;
  }

  long double ExcessKurtosis() const override
  {
      return -1.2 / n;
  }

private:
  std::complex<double> CFImpl(double t) const override
  {
      return std::pow(U.CF(t), n);
  }
};
} // namespace randlib
