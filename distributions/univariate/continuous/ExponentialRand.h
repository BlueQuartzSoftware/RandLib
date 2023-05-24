#ifndef EXPONENTIALRAND_H
#define EXPONENTIALRAND_H

#include "GammaRand.h"

#include "external/log.hpp"

#include <array>
#include <functional>

/**
 * @brief The ExponentialRand class <BR>
 * Exponential distribution
 *
 * f(x | β) = β exp(-βx)
 *
 * Notation: X ~ Exp(β)
 *
 * Related distributions: <BR>
 * X ~ Γ(1, β)
 */
template <typename RealType = double>
class RANDLIB_EXPORT ExponentialRand : public FreeRateGammaDistribution<RealType>, public ExponentialFamily<RealType, double>
{
private:
  template <typename T = void>
  struct ExpZiggurat
  {
    struct Ziggurat
    {
      std::array<LongDoublePair, 257> table = {};

      constexpr Ziggurat()
      {
        constexpr long double A = 3.9496598225815571993e-3l; /// area under rectangle
        // coordinates of the implicit rectangle in base layer
        table[0].first = 0.00045413435384149675l;   /// exp(-x1);
        table[0].second = 8.697117470131049720307l; /// A / stairHeight[0];
        /// implicit value for the top layer
        table[257 - 1].second = 0;
        table[1].second = 7.69711747013104972l;
        table[1].first = 0.0009672692823271745203l;
        for(size_t i = 2; i < 257 - 1; ++i)
        {
          /// such y_i that f(x_{i+1}) = y_i
          table[i].second = -nonstd::log(table[i - 1].first);
          table[i].first = table[i - 1].first + A / table[i].second;
        }
      }
    };
    static constexpr Ziggurat STATIC_ZIGGURAT = Ziggurat();
  };

  static constexpr auto ziggurat = ExpZiggurat<>::STATIC_ZIGGURAT.table;

public:
  explicit ExponentialRand(double rate = 1)
  : FreeRateGammaDistribution<RealType>(1, rate)
  {
  }
  String Name() const override;

  double SufficientStatistic(RealType x) const override;
  double SourceParameters() const override;
  double SourceToNatural(double rate) const override;
  double LogNormalizer(double theta) const override;
  double LogNormalizerGradient(double theta) const override;
  double CarrierMeasure(RealType) const override;
  double CrossEntropyAdjusted(double rate) const override;
  double EntropyAdjusted() const override;

  double f(const RealType& x) const override;
  double logf(const RealType& x) const override;
  double F(const RealType& x) const override;
  double S(const RealType& x) const override;
  RealType Variate() const override;
  void Sample(std::vector<RealType>& outputData) const override;
  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator);

private:
  std::complex<double> CFImpl(double t) const override;

public:
  long double Entropy() const;
  long double Moment(int n) const;
  long double ThirdMoment() const override
  {
    return Moment(3);
  }
  long double FourthMoment() const override
  {
    return Moment(4);
  }
};

#endif // EXPONENTIALRAND_H
