#pragma once

#include "distributions/univariate/continuous/GammaRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"

#include "external/log.hpp"

#include <array>
#include <functional>

namespace randlib
{
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

  std::complex<double> CFImpl(double t) const override
  {
      return 1.0 / std::complex<double>(1.0, -this->theta * t);
  }

public:
  explicit ExponentialRand(double rate = 1)
  : FreeRateGammaDistribution<RealType>(1, rate)
  {
  }

  String Name() const override
  {
      return "Exponential(" + this->toStringWithPrecision(this->GetRate()) + ")";
  }

  double SufficientStatistic(RealType x) const override
  {
      return x;
  }

  double SourceParameters() const override
  {
      return this->beta;
  }

    double SourceToNatural(double rate) const override
    {
        return -rate;
    }

  double LogNormalizer(double theta) const override
  {
      return -std::log(-theta);
  }

    double LogNormalizerGradient(double theta) const override
    {
        return -1.0 / theta;
    }

  double CarrierMeasure(RealType) const override
  {
      return 0.0;
  }

  double CrossEntropyAdjusted(double rate) const override
  {
      return rate * this->theta - std::log(rate);
  }

  double EntropyAdjusted() const override
  {
      return 1.0 - this->logBeta;
  }

  double f(const RealType& x) const override
  {
      return (x < 0.0) ? 0.0 : this->beta * std::exp(-this->beta * x);
  }

  double logf(const RealType& x) const override
  {
      return (x < 0.0) ? -INFINITY : this->logBeta - this->beta * x;
  }

    double F(const RealType& x) const override
    {
        return (x > 0.0) ? -std::expm1l(-this->beta * x) : 0.0;
    }

  double S(const RealType& x) const override
  {
      return (x > 0.0) ? std::exp(-this->beta * x) : 1.0;
  }

    RealType Variate() const override
    {
        return this->theta * StandardVariate(this->localRandGenerator);
    }

  void Sample(std::vector<RealType>& outputData) const override
  {
      for(RealType& var : outputData)
          var = this->Variate();
  }

  static RealType StandardVariate(RandGenerator& randGenerator = ProbabilityDistribution<RealType>::staticRandGenerator)
  {
      /// Ziggurat algorithm
      size_t iter = 0;
      do
      {
          int stairId = randGenerator.Variate() & 255;
          /// Get horizontal coordinate
          RealType x = UniformRand<RealType>::StandardVariate(randGenerator) * ziggurat[stairId].second;
          if(x < ziggurat[stairId + 1].second) /// if we are under the upper stair - accept
              return x;
          if(stairId == 0) /// if we catch the tail
              return ziggurat[1].second + StandardVariate(randGenerator);
          RealType height = ziggurat[stairId].first - ziggurat[stairId - 1].first;
          if(ziggurat[stairId - 1].first + height * UniformRand<RealType>::StandardVariate(randGenerator) < std::exp(-x)) /// if we are under the curve - accept
              return x;
          /// rejection - go back
      } while(++iter <= ProbabilityDistribution<RealType>::MAX_ITER_REJECTION);
      /// fail due to some error
      throw std::runtime_error("Exponential distribution: sampling failed");
  }

  long double Entropy() const
  {
      return this->EntropyAdjusted();
  }

  long double Moment(int n) const
  {
      if(n < 0)
          return 0;
      if(n == 0)
          return 1;
      return std::exp(RandMath::lfact(n) - n * this->logBeta);
  }

  long double ThirdMoment() const override
  {
    return Moment(3);
  }

  long double FourthMoment() const override
  {
    return Moment(4);
  }
};
} // namespace randlib