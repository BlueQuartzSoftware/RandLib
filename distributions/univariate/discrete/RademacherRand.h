#ifndef RADEMACHERRAND_H
#define RADEMACHERRAND_H

#include "distributions/Distributions.h"
#include "distributions/univariate/BasicRandGenerator.h"

/**
 * @brief The RademacherRand class <BR>
 * Rademacher distribution
 *
 * P(X = k) = 0.5 * 1_{|k| = 1}
 *
 * Notation: X ~ Rademacher
 *
 * Related distributions: <BR>
 * If Y ~ Bernoulli(0.5), then 2Y - 1 ~ Rademacher
 */
class RANDLIB_EXPORT RademacherRand : public distributions::DiscreteDistribution<int>
{
public:
  RademacherRand();
  String Name() const override;
  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }
  int MinValue() const override
  {
    return -1;
  }
  int MaxValue() const override
  {
    return 1;
  }

  double P(const int& k) const override;
  double logP(const int& k) const override;
  double F(const int& k) const override;
  int Variate() const override;

  long double Mean() const override;
  long double Variance() const override;
  int Median() const override;
  int Mode() const override;
  long double Skewness() const override;
  long double ExcessKurtosis() const override;

private:
  int quantileImpl(double p) const override;
  int quantileImpl1m(double p) const override;
  std::complex<double> CFImpl(double t) const override;

public:
  double Entropy();
};

#endif // RADEMACHERRAND_H
