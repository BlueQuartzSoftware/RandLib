#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/ContinuousDistributions.hpp"

#include "distributions/univariate/continuous/UniformRand.hpp"

namespace randlib
{
/**
 * @brief The RaisedCosineDistribution class <BR>
 * Abstract class for Raised-cosine distribution
 *
 * Notation: X ~ Raised-cosine(μ, s)
 */
template <typename RealType = double>
class RANDLIB_EXPORT RaisedCosineDistribution : public randlib::ContinuousDistribution<RealType>
{
  double mu = 0;                 ///< location μ
  double s = M_PI;               ///< scale
  double s_pi = 1;               ///< s / π
  double log2S = M_LN2 + M_LNPI; ///< log(2s)

protected:
  RaisedCosineDistribution(double location, double scale)
  {
      SetLocation(location);
      SetScale(scale);
  }

public:
  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  RealType MinValue() const override
  {
    return mu - s;
  }

  RealType MaxValue() const override
  {
    return mu + s;
  }

  inline double GetLocation() const
  {
    return mu;
  }

  inline double GetScale() const
  {
    return s;
  }

  double f(const RealType& x) const override
  {
      double xAdj = (x - mu) / s_pi;
      if(xAdj <= -M_PI || xAdj >= M_PI)
          return 0.0;
      double y = std::cos(xAdj) + 1.0;
      return 0.5 * y / s;
  }

  double logf(const RealType& x) const override
  {
      double xAdj = (x - mu) / s_pi;
      if(xAdj <= -M_PI || xAdj >= M_PI)
          return -INFINITY;
      double y = std::cos(xAdj);
      y = std::log1pl(y);
      return y - log2S;
  }

  double F(const RealType& x) const override
  {
      double xAdj = (x - mu) / s;
      if(xAdj <= -1)
          return 0.0;
      if(xAdj >= 1)
          return 1.0;
      double y = std::sin(xAdj * M_PI);
      y *= M_1_PI;
      y += xAdj + 1;
      return 0.5 * y;
  }

  double S(const RealType& x) const override
  {
      double xAdj = (x - mu) / s;
      if(xAdj <= -1)
          return 1.0;
      if(xAdj >= 1)
          return 0.0;
      double y = std::sin(xAdj * M_PI);
      y *= M_1_PI;
      y += xAdj;
      return 0.5 - 0.5 * y;
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      /// p. 160. Non-Uniform Random Variate Generation. Luc Devroye
      RealType X = M_PI * randlib::UniformRand<RealType>::StandardVariate(randGenerator) - M_PI_2;
      RealType XSq = X * X;
      RealType U = 2 * randlib::UniformRand<RealType>::StandardVariate(randGenerator);
      int a = 0, b = -1;
      RealType W = 0.0, V = 1.0;
      size_t iter = 0;
      do
      {
          a += 2;
          b += 2;
          V *= XSq / (a * b);
          W += V;
          if(U >= W)
              return X;
          a += 2;
          b += 2;
          V *= XSq / (a * b);
          W -= V;
          if(U <= W)
          {
              if(X == 0.0)
                  return 0.0;
              return (X > 0.0) ? M_PI - X : -M_PI - X;
          }
      } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
      throw std::runtime_error("Raised-Cosine distribution: sampling failed");
  }

  RealType Variate() const override
  {
      return mu + s_pi * StandardVariate(this->localRandGenerator);
  }

  long double Mean() const override
  {
      return mu;
  }

  long double Variance() const override
  {
      static constexpr double coef = 1.0 / 3 - 2.0 / M_PI_SQ;
      return s * s * coef;
  }

  RealType Median() const override
  {
      return mu;
  }

  RealType Mode() const override
  {
      return mu;
  }

  long double Skewness() const override
  {
      return 0.0l;
  }

  long double ExcessKurtosis() const override
  {
      static constexpr long double numerator = 1.2 * (90.0 - M_PI_SQ * M_PI_SQ);
      static constexpr long double denominator = M_PI_SQ - 6.0;
      static constexpr long double y = numerator / (denominator * denominator);
      return y;
  }

protected:
    void SetLocation(double location)
    {
        mu = location;
    }

    void SetScale(double scale)
    {
        if(scale <= 0.0)
            throw std::invalid_argument("Raised-Cosine distribution: scale should be positive");
        s = scale;
        s_pi = s * M_1_PI;
        log2S = std::log(2 * s);
    }

private:
  std::complex<double> CFImpl(double t) const override
  {
      double st = s * t;
      double numerator = M_PI_SQ * std::sin(st);
      double denominator = st * (M_PI_SQ - st * st);
      std::complex<double> y(0.0, mu * t);
      y = std::exp(y);
      return numerator / denominator * y;
  }
};

/**
 * @brief The RaisedCosineRand class <BR>
 * Raised-cosine distribution
 */
template <typename RealType = double>
class RANDLIB_EXPORT RaisedCosineRand : public RaisedCosineDistribution<RealType>
{
public:
  RaisedCosineRand(double location = 0, double scale = M_PI)
  : RaisedCosineDistribution<RealType>(location, scale)
  {
  }

  String Name() const override
  {
      return "Raised cosine(" + this->toStringWithPrecision(this->GetLocation()) + ", " + this->toStringWithPrecision(this->GetScale()) + ")";
  }

  using RaisedCosineDistribution<RealType>::SetLocation;
  using RaisedCosineDistribution<RealType>::SetScale;
};

/**
 * @brief The RaabGreenRand class <BR>
 * Raab-Green distribution
 *
 * Notation: X ~ Raab-Green()
 *
 * Related distributions:
 * X ~ Raised-cosine(0.0, π)
 */
template <typename RealType = double>
class RANDLIB_EXPORT RaabGreenRand : public RaisedCosineDistribution<RealType>
{
public:
  RaabGreenRand()
  : RaisedCosineDistribution<RealType>(0.0, M_PI)
  {
  }

  String Name() const override
  {
      return "Raab Green";
  }
};
} // namespace randlib
