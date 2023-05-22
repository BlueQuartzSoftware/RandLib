#ifndef PLANCKRAND_H
#define PLANCKRAND_H

#include "Distributions.h"
#include "GammaRand.h"
#include "univariate/discrete/ZetaRand.h"

/**
 * @brief The PlanckRand class <BR>
 * Planck distribution
 *
 * f(x | a, b) = g(a + 1) * (x ^ a) / (exp(b * x) - 1)
 * where g(y) = b ^ y / (Γ(y) * ζ(y))
 *
 * Notation: X ~ Planck(a, b)
 *
 * Related distributions: <BR>
 * If G ~ Gamma(a + 1, b) and Z ~ Zeta(a + 1), then G / Z ~ Planck(a, b)
 */
template <typename RealType = double>
class RANDLIBSHARED_EXPORT PlanckRand
    : public distributions::ContinuousDistribution<RealType> {
  double a = 1; ///< shape
  double b = 1; ///< scale
  double pdfCoef =
      M_LN2 + M_LN3 - 2 * M_LNPI; ///< coefficient for faster pdf calculations

  ZetaRand<long long int> Z{};
  GammaRand<RealType> G{2};

public:
  PlanckRand(double shape = 1, double scale = 1);

  String Name() const override;
  SUPPORT_TYPE SupportType() const override {
    return SUPPORT_TYPE::RIGHTSEMIFINITE_T;
  }
  RealType MinValue() const override { return 0; }
  RealType MaxValue() const override { return INFINITY; }

  void SetParameters(double shape, double scale);
  inline double GetShape() const { return a; }
  inline double GetScale() const { return b; }

private:
  /**
   * @fn h
   * @param t
   * @return leveled pdf for F and CF calculations
   */
  double h(double t) const;

public:
  double f(const RealType &x) const override;
  double logf(const RealType &x) const override;
  double F(const RealType &x) const override;
  RealType Variate() const override;
  void Sample(std::vector<RealType> &outputData) const override;

  long double Mean() const override;
  long double SecondMoment() const override;
  long double Variance() const override;
  RealType Mode() const override;
  long double ThirdMoment() const override;
  long double Skewness() const override;
  long double FourthMoment() const override;
  long double ExcessKurtosis() const override;

private:
  std::complex<double> CFImpl(double t) const override;
};

#endif // PLANCKRAND_H
