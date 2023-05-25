#pragma once

#include "RandLib_export.hpp"
#include "RandLib.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/BinomialRand.hpp"

namespace randlib
{
/**
 * @brief The TrinomialRand class <BR>
 * Trinomial distribution
 *
 * Notation: X ~ Trin(n, p_1, p_2)
 *
 * Related distributions: <BR>
 * X ~ Multin(n, 1 - p_1 - p_2, p_1, p_2)
 */
template <typename IntType = int>
class RANDLIB_EXPORT TrinomialRand : public randlib::DiscreteBivariateDistribution<BinomialRand<IntType>, BinomialRand<IntType>, IntType>
{
  int n = 1;                 ///< number of trials
  double log1mProb = -M_LN3; ///< log(1 - p_1 - p_2)
  double p1_1mp2 = 0.5;      ///< p_1 / (1 - p_2)
  double p2_1mp1 = 0.5;      ///< p_2 / (1 - p_1)
public:
  TrinomialRand(int number, double probability1, double probability2)
  {
    SetParameters(number, probability1, probability2);
  }

  String Name() const override
  {
    return "Trinomial(" + this->toStringWithPrecision(GetNumber()) + ", " + this->toStringWithPrecision(GetFirstProbability()) + ", " + this->toStringWithPrecision(GetSecondProbability()) + ")";
  }

  void SetParameters(int number, double probability1, double probability2)
  {
    if(probability1 < 0.0 || probability2 < 0.0 || probability1 + probability2 > 1.0)
      throw std::invalid_argument("Probabilities of Trinomial distribution should be positive and their "
                                  "sum should be not greater than 1, but they're equal to " +
                                  std::to_string(probability1) + " and " + std::to_string(probability2));
    if(number <= 0)
      throw std::invalid_argument("Number of Trinomial distribution should be "
                                  "positive, but it's equal to " +
                                  std::to_string(number));

    double p1 = probability1, p2 = probability2;
    n = number;
    this->X.SetParameters(n, p1);
    this->Y.SetParameters(n, p2);

    log1mProb = std::log1pl(-p1 - p2);
    p1_1mp2 = p1 / (1.0 - p2);
    p2_1mp1 = p2 / (1.0 - p1);
  }

  inline int GetNumber() const
  {
    return n;
  }

  inline double GetFirstProbability() const
  {
    return this->X.GetProbability();
  }

  inline double GetSecondProbability() const
  {
    return this->Y.GetProbability();
  }

  double P(const Pair<IntType>& point) const override
  {
    IntType x = point.first, y = point.second;
    return (x < 0 || x > n || y < 0 || y > n) ? 0.0 : std::exp(logP(point));
  }

  double logP(const Pair<IntType>& point) const override
  {
    IntType x = point.first, y = point.second;
    if(x < 0 || y < 0 || x + y > n)
      return -INFINITY;
    IntType nmxmy = n - x - y;
    double res = this->X.GetLogFactorialN();
    res -= RandMath::lfact(x);
    res -= RandMath::lfact(y);
    res -= RandMath::lfact(nmxmy);
    res += x * this->X.GetLogProbability();
    res += y * this->Y.GetLogProbability();
    res += nmxmy * log1mProb;
    return res;
  }

  double F(const Pair<IntType>& point) const override
  {
    int x = point.first, y = point.second;
    if(x < 0 || y < 0)
      return 0.0;
    if(x >= n)
      return this->Y.F(y);
    if(y >= n)
      return this->X.F(x);
    double sum = 0.0;
    double p1 = this->X.GetProbability(), p2 = this->Y.GetProbability();
    if(x > y)
    {
      double ratio = (1 - p2 - p1) / (1 - p2);
      for(int i = 0; i <= y; ++i)
      {
        int nmimx = n - i - x;
        double prob1 = (nmimx > 0) ? RandMath::ibeta(ratio, nmimx, x + 1) : 1.0;
        double prob2 = this->Y.P(i);
        sum += prob1 * prob2;
      }
      return sum;
    }
    double ratio = (1 - p1 - p2) / (1 - p1);
    for(int i = 0; i <= x; ++i)
    {
      IntType nmimy = n - i - y;
      double prob1 = (nmimy > 0) ? RandMath::ibeta(ratio, nmimy, y + 1) : 1.0;
      double prob2 = this->X.P(i);
      sum += prob1 * prob2;
    }
    return sum;
  }

  Pair<IntType> Variate() const override
  {
    if(this->X.GetProbability() > this->Y.GetProbability())
    {
      IntType x = this->X.Variate();
      IntType y = BinomialDistribution<IntType>::Variate(n - x, p2_1mp1, this->localRandGenerator);
      return IntPair(x, y);
    }
    IntType y = this->Y.Variate();
    IntType x = BinomialDistribution<IntType>::Variate(n - y, p1_1mp2, this->localRandGenerator);
    return IntPair(x, y);
  }

  long double Correlation() const override
  {
    long double logp1 = this->X.GetLogProbability(), logp2 = this->Y.GetLogProbability();
    long double log1mp1 = this->X.GetLog1mProbability(), log1mp2 = this->Y.GetLog1mProbability();
    return -std::exp(logp1 + logp2 - log1mp1 - log1mp2);
  }

  Pair<IntType> Mode() const override
  {
    IntType modeX = this->X.Mode(), modeY = this->Y.Mode();
    double modeValue = this->P(std::make_pair(modeX, modeY));
    double modeValuexp1 = this->P(std::make_pair(modeX + 1, modeY));
    double modeValuexm1 = this->P(std::make_pair(modeX - 1, modeY));
    double modeValueyp1 = this->P(std::make_pair(modeX, modeY + 1));
    double modeValueym1 = this->P(std::make_pair(modeX, modeY - 1));

    while(modeValuexp1 > modeValue)
    {
      modeX += 1;
      modeValue = modeValuexp1;
      modeValuexp1 = this->P(std::make_pair(modeX + 1, modeY));
    }

    while(modeValuexm1 > modeValue)
    {
      modeX -= 1;
      modeValue = modeValuexm1;
      modeValuexp1 = this->P(std::make_pair(modeX - 1, modeY));
    }

    while(modeValueyp1 > modeValue)
    {
      modeY += 1;
      modeValue = modeValueyp1;
      modeValueyp1 = this->P(std::make_pair(modeX, modeY + 1));
    }

    while(modeValueym1 > modeValue)
    {
      modeY += 1;
      modeValue = modeValueym1;
      modeValueym1 = this->P(std::make_pair(modeX, modeY - 1));
    }

    return std::make_pair(modeX, modeY);
  }
};
} // namespace randlib
