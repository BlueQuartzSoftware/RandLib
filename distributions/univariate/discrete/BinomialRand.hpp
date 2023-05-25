#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

#include "distributions/DiscreteDistributions.hpp"

#include "distributions/univariate/discrete/BernoulliRand.hpp"
#include "distributions/univariate/continuous/BetaRand.hpp"
#include "distributions/univariate/continuous/ExponentialRand.hpp"
#include "distributions/univariate/continuous/NormalRand.hpp"
#include "distributions/univariate/continuous/UniformRand.hpp"
#include "distributions/univariate/discrete/GeometricRand.hpp"

namespace randlib
{
/**
 * @brief The BinomialDistribution class <BR>
 * Abstract class for Binomial distribution
 *
 * Notation: X ~ Bin(n, p)
 *
 * Related distributions: <BR>
 * If X ~ Bin(1, p), then X ~ Bernoulli(p) <BR>
 * X ~ Multin(n, 1 - p, p)
 */
template <typename IntType = int>
class RANDLIB_EXPORT BinomialDistribution : public randlib::DiscreteDistribution<IntType>, public ExponentialFamily<IntType, double>
{
protected:
  double p = 0.5;            ///< probability of success
  double q = 0.5;            ///< probability of failure
  double logProb = -M_LN2;   ///< log(p)
  double log1mProb = -M_LN2; ///< log(q)

  BinomialDistribution(IntType number, double probability)
  {
    SetParameters(number, probability);
  }

public:
  SUPPORT_TYPE SupportType() const override
  {
    return SUPPORT_TYPE::FINITE_T;
  }

  IntType MinValue() const override
  {
    return 0;
  }

  IntType MaxValue() const override
  {
    return n;
  }

  inline IntType GetNumber() const
  {
    return n;
  }

  inline double GetProbability() const
  {
    return p;
  }

  double SufficientStatistic(IntType x) const override
  {
    return x;
  }

  double SourceParameters() const override
  {
    return this->p;
  }

  double SourceToNatural(double sourceParameters) const override
  {
    return std::log(sourceParameters) - std::log1p(-sourceParameters);
  }

  double NaturalParameters() const override
  {
    return this->logProb - this->log1mProb;
  }

  double LogNormalizer(double theta) const override
  {
    double F = n * RandMath::softplus(theta);
    F -= this->lfactn;
    return F;
  }

  double LogNormalizerGradient(double theta) const override
  {
    double expTheta = std::exp(theta);
    return this->n * expTheta / (1.0 + expTheta);
  }

  double CarrierMeasure(IntType x) const override
  {
    double k = -RandMath::lfact(x);
    k -= RandMath::lfact(n - x);
    return k;
  }

  double EntropyAdjusted() const override
  {
    double H = this->p * this->logProb;
    H += this->q * this->log1mProb;
    H *= this->n;
    H += this->lfactn;
    return -H;
  }

  double P(const IntType& k) const override
  {
    return (k < 0 || k > n) ? 0.0 : std::exp(logP(k));
  }

  double logP(const IntType& k) const override
  {
    if(k < 0 || k > n)
      return -INFINITY;
    double y = lfactn;
    y -= RandMath::lfact(n - k);
    y -= RandMath::lfact(k);
    y += k * logProb;
    y += (n - k) * log1mProb;
    return y;
  }

  double F(const IntType& k) const override
  {
    if(k < 0)
      return 0.0;
    if(k >= n)
      return 1.0;
    int nmk = n - k, kp1 = k + 1;
    double logBetaFun = RandMath::lfact(n - kp1);
    logBetaFun += RandMath::lfact(k);
    logBetaFun -= lfactn;
    return RandMath::ibeta(q, nmk, kp1, logBetaFun, log1mProb, logProb);
  }

  double S(const IntType& k) const override
  {
    if(k < 0)
      return 1.0;
    if(k >= n)
      return 0.0;
    int nmk = n - k, kp1 = k + 1;
    double logBetaFun = RandMath::logBeta(kp1, nmk);
    return RandMath::ibeta(p, kp1, nmk, logBetaFun, logProb, log1mProb);
  }

  IntType Variate() const override
  {
    GENERATOR_ID genId = GetIdOfUsedGenerator();
    switch(genId)
    {
    case WAITING: {
      IntType var = variateWaiting(n);
      return (p <= 0.5) ? var : n - var;
    }
    case REJECTION: {
      /// if X ~ Bin(n, p') and Y ~ Bin(n - X, (p - p') / (1 - p'))
      /// then Z = X + Y ~ Bin(n, p)
      IntType Z = variateRejection();
      if(pRes > 0)
        Z += variateWaiting(n - Z);
      return (p > 0.5) ? n - Z : Z;
    }
    case BERNOULLI_SUM:
      return variateBernoulliSum(n, p, this->localRandGenerator);
    default:
      throw std::invalid_argument("Binomial distribution: invalid generator id");
    }
  }

  static IntType Variate(IntType number, double probability, RandGenerator& randGenerator = ProbabilityDistribution<IntType>::staticRandGenerator)
  {
    /// sanity check
    if(number < 0)
      throw std::invalid_argument("Binomial distribution: number should be positive, but it's equal to " + std::to_string(number));
    if(probability < 0.0 || probability > 1.0)
      throw std::invalid_argument("Binomial distribution: probability parameter should in interval [0, "
                                  "1], but it's equal to " +
                                  std::to_string(probability));
    if(probability == 0.0)
      return 0;
    if(probability == 1.0)
      return number;

    if(number < 10)
      return variateBernoulliSum(number, probability, randGenerator);
    if(probability < 0.5)
      return variateWaiting(number, probability, randGenerator);
    return number - variateWaiting(number, 1.0 - probability, randGenerator);
  }

  void Sample(std::vector<IntType>& outputData) const override
  {
    if(p == 0.0)
    {
      std::fill(outputData.begin(), outputData.end(), 0);
      return;
    }
    if(RandMath::areClose(p, 1.0))
    {
      std::fill(outputData.begin(), outputData.end(), n);
      return;
    }

    GENERATOR_ID genId = GetIdOfUsedGenerator();
    switch(genId)
    {
    case WAITING: {
      if(p <= 0.5)
      {
        for(IntType& var : outputData)
          var = variateWaiting(n);
      }
      else
      {
        for(IntType& var : outputData)
          var = n - variateWaiting(n);
      }
      return;
    }
    case REJECTION: {
      for(IntType& var : outputData)
        var = variateRejection();
      if(pRes > 0)
      {
        for(IntType& var : outputData)
          var += variateWaiting(n - var);
      }
      if(p > 0.5)
      {
        for(IntType& var : outputData)
          var = n - var;
      }
      return;
    }
    case BERNOULLI_SUM:
    default: {
      for(IntType& var : outputData)
        var = variateBernoulliSum(n, p, this->localRandGenerator);
      return;
    }
    }
  }

  void Reseed(unsigned long seed) const override
  {
    this->localRandGenerator.Reseed(seed);
    G.Reseed(seed);
  }

  long double Mean() const override
  {
    return np;
  }

  long double Variance() const override
  {
    return np * q;
  }

  IntType Median() const override
  {
    return std::floor(np);
  }

  IntType Mode() const override
  {
    return std::floor(np + p);
  }

  long double Skewness() const override
  {
    return (q - p) / std::sqrt(np * q);
  }

  long double ExcessKurtosis() const override
  {
    long double y = 1.0 / (p * q);
    y -= 6.0;
    return y / n;
  }

  /**
   * @fn GetLogFactorialN
   * @return log(n!)
   */
  inline double GetLogFactorialN() const
  {
    return lfactn;
  }

  /**
   * @fn GetLogProbability
   * @return log(p)
   */
  inline double GetLogProbability() const
  {
    return logProb;
  }

  /**
   * @fn GetLog1mProbability
   * @return log(1-p)
   */
  inline double GetLog1mProbability() const
  {
    return log1mProb;
  }

  /**
   * @fn FitProbability
   * Fit probability p with maximum-likelihood estimation
   * @param sample
   */
  void FitProbability(const std::vector<IntType>& sample)
  {
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    if(!this->allElementsAreNotGreaterThan(n, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(n)));
    SetParameters(n, this->GetSampleMean(sample) / n);
  }

  /**
   * @fn FitProbabilityBayes
   * Fit probability p with prior assumption p ~ Beta(α, β)
   * @param sample
   * @param priorDistribution
   * @param MAP if true, use MAP estimator
   * @return posterior distribution
   */
  BetaRand<> FitProbabilityBayes(const std::vector<IntType>& sample, const BetaDistribution<>& priorDistribution, bool MAP = false)
  {
    if(!this->allElementsAreNonNegative(sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->NON_NEGATIVITY_VIOLATION));
    if(!this->allElementsAreNotGreaterThan(n, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(n)));
    int N = sample.size();
    double sum = this->GetSampleSum(sample);
    double alpha = priorDistribution.GetAlpha();
    double beta = priorDistribution.GetBeta();
    BetaRand posteriorDistribution(sum + alpha, N * n - sum + beta);
    SetParameters(n, MAP ? posteriorDistribution.Mode() : posteriorDistribution.Mean());
    return posteriorDistribution;
  }

  /**
   * @fn FitProbabilityMinimax
   * Fit probability p with minimax estimator
   * @param sample
   * @return posterior distribution
   */
  BetaRand<> FitProbabilityMinimax(const std::vector<IntType>& sample)
  {
    double shape = 0.5 * std::sqrt(n);
    BetaRand B(shape, shape);
    return FitProbabilityBayes(sample, B);
  }

protected:
  void SetParameters(IntType number, double probability)
  {
    if(probability < 0.0 || probability > 1.0)
      throw std::invalid_argument("Binomial distribution: probability parameter "
                                  "should in interval [0, 1]");
    if(number <= 0)
      throw std::invalid_argument("Binomial distribution: number should be positive");
    n = number;
    p = probability;
    q = 1.0 - p;
    np = n * p;
    lfactn = RandMath::lfact(n);
    logProb = std::log(p);
    log1mProb = std::log1pl(-p);
    SetGeneratorConstants();
  }

private:
  IntType n = 1;     ///< number of experiments
  double np = 0.5;   ///< n * p
  double lfactn = 0; ///< log(n!)

  double delta1{}, delta2{};
  double sigma1{}, sigma2{}, c{};
  double a1{}, a2{}, a3{}, a4{};
  double coefa3{}, coefa4{};

  double minpq = 0.5;           ///< min(p, q)
  double pFloor = 0;            ///< [n * min(p, q)] / n
  double logPFloor = -INFINITY; ///< log(pFloor)
  double logQFloor = 0;         ///< log(1 - pFloor)
  double pRes = 0.5;            ///< min(p, q) - pFloor
  double npFloor = 0;           ///< [n * min(p, q)]
  double nqFloor = 0;           ///< [n * max(p, q)]
  double logPnpInv = 0;         ///< log(P([npFloor)) if p = pFloor

  GeometricRand<IntType> G{};

  void SetGeneratorConstants()
  {
    minpq = std::min(p, q);
    npFloor = std::floor(n * minpq);
    pFloor = npFloor / n;
    pRes = RandMath::areClose(npFloor, n * minpq) ? 0.0 : minpq - pFloor;

    GENERATOR_ID genId = GetIdOfUsedGenerator();
    if(genId == BERNOULLI_SUM)
      return;
    else if(genId == WAITING)
    {
      G.SetProbability(minpq);
      return;
    }

    nqFloor = n - npFloor;
    double qFloor = 1.0 - pFloor;
    if(pRes > 0)
      G.SetProbability(pRes / qFloor);

    /// Set deltas
    double npq = npFloor * qFloor;
    double coef = 128.0 * n / M_PI;
    delta1 = coef * pFloor / (81.0 * qFloor);
    delta1 = npq * std::log(delta1);
    if(delta1 > 1.0)
      delta1 = std::sqrt(delta1);
    else
      delta1 = 1.0;
    delta2 = coef * qFloor / pFloor;
    delta2 = npq * std::log(delta2);
    if(delta2 > 1.0)
      delta2 = std::sqrt(delta2);
    else
      delta2 = 1.0;

    /// Set sigmas and c
    double npqSqrt = std::sqrt(npq);
    sigma1 = npqSqrt * (1.0 + 0.25 * delta1 / npFloor);
    sigma2 = npqSqrt * (1.0 + 0.25 * delta2 / nqFloor);
    c = 2.0 * delta1 / npFloor;

    /// Set a's
    a1 = 0.5 * std::exp(c) * sigma1 * M_SQRT2PI;

    a2 = 0.5 * sigma2 * M_SQRT2PI;
    a2 += a1;

    coefa3 = 0.5 * delta1 / (sigma1 * sigma1);
    a3 = 1.0 / nqFloor - coefa3;
    a3 *= delta1;
    a3 = std::exp(a3);
    a3 /= coefa3;
    a3 += a2;

    coefa4 = 0.5 * delta2 / (sigma2 * sigma2);
    a4 = std::exp(-delta2 * coefa4) / coefa4;
    a4 += a3;

    logPFloor = std::log(pFloor);
    logQFloor = (pFloor == qFloor) ? logPFloor : std::log1pl(-pFloor);

    logPnpInv = logProbFloor(npFloor);
  }

  /**
   * @fn logProbFloor
   * @param k
   * @return logarithm of probability to get k if p = pFloor
   */
  double logProbFloor(int k) const
  {
    double y = lfactn;
    y -= RandMath::lfact(n - k);
    y -= RandMath::lfact(k);
    y += k * logPFloor;
    y += (n - k) * logQFloor;
    return y;
  }

  enum GENERATOR_ID
  {
    BERNOULLI_SUM,
    WAITING,
    REJECTION,
    POISSON
  };

  GENERATOR_ID GetIdOfUsedGenerator() const
  {
    /// if (n is tiny and minpq is big) or (p ~= 0.5 and n is not that large),
    /// we just sum Bernoulli random variables
    if((n <= 3) || (n <= 13 && minpq > 0.025 * (n + 6)) || (n <= 200 && RandMath::areClose(p, 0.5)))
      return BERNOULLI_SUM;

    /// for small [np] we use simple waiting algorithm
    if((npFloor <= 12) || (pRes > 0 && npFloor <= 16))
      return WAITING;

    /// otherwise
    return REJECTION;
  }

  IntType variateRejection() const
  {
    /// a rejection algorithm by Devroye and Naderlsamanl (1980)
    /// p.533. Non-Uniform Random Variate Generation. Luc Devroye
    /// it can be used only when n * p is integer and p < 0.5
    bool reject = true;
    size_t iter = 0;
    float Y, V;
    IntType X;
    do
    {
      float U = a4 * UniformRand<float>::StandardVariate(this->localRandGenerator);
      if(U <= a1)
      {
        float N = NormalRand<float>::StandardVariate(this->localRandGenerator);
        Y = sigma1 * std::fabs(N);
        reject = (Y >= delta1);
        if(!reject)
        {
          float W = randlib::ExponentialRand<float>::StandardVariate(this->localRandGenerator);
          X = std::floor(Y);
          V = -W - 0.5 * N * N + c;
        }
      }
      else if(U <= a2)
      {
        float N = NormalRand<float>::StandardVariate(this->localRandGenerator);
        Y = sigma2 * std::fabs(N);
        reject = (Y >= delta2);
        if(!reject)
        {
          float W = ExponentialRand<float>::StandardVariate(this->localRandGenerator);
          X = std::floor(-Y);
          V = -W - 0.5 * N * N;
        }
      }
      else if(U <= a3)
      {
        float W1 = ExponentialRand<float>::StandardVariate(this->localRandGenerator);
        float W2 = ExponentialRand<float>::StandardVariate(this->localRandGenerator);
        Y = delta1 + W1 / coefa3;
        X = std::floor(Y);
        V = -W2 - coefa3 * Y + delta1 / nqFloor;
        reject = false;
      }
      else
      {
        float W1 = ExponentialRand<float>::StandardVariate(this->localRandGenerator);
        float W2 = ExponentialRand<float>::StandardVariate(this->localRandGenerator);
        Y = delta2 + W1 / coefa4;
        X = std::floor(-Y);
        V = -W2 - coefa4 * Y;
        reject = false;
      }

      if(!reject)
      {
        X += npFloor;
        if(X >= 0 && X <= n && V <= logProbFloor(X) - logPnpInv)
          return X;
      }
    } while(++iter <= ProbabilityDistribution<IntType>::MAX_ITER_REJECTION);
    throw std::runtime_error("Binomial distribution: sampling failed");
  }

  IntType variateWaiting(IntType number) const
  {
    /// waiting algorithm, using
    /// sum of geometrically distributed variables
    IntType X = -1, sum = 0;
    do
    {
      sum += G.Variate() + 1;
      ++X;
    } while(sum <= number);
    return X;
  }

  static IntType variateWaiting(IntType number, double probability, RandGenerator& randGenerator)
  {
    IntType X = -1;
    double sum = 0;
    do
    {
      IntType add = GeometricRand<IntType>::Variate(probability, randGenerator) + 1;
      if(add < 0) /// we catched overflow
        return X + 1;
      sum += add;
      ++X;
    } while(sum <= number);
    return X;
  }

  static IntType variateBernoulliSum(IntType number, double probability, RandGenerator& randGenerator)
  {
    IntType var = 0;
    if(RandMath::areClose(probability, 0.5))
    {
      for(int i = 0; i != number; ++i)
        var += randlib::BernoulliRand::StandardVariate(randGenerator);
    }
    else
    {
      for(int i = 0; i != number; ++i)
        var += randlib::BernoulliRand::Variate(probability, randGenerator);
    }
    return var;
  }

  std::complex<double> CFImpl(double t) const override
  {
    std::complex<double> y(q + p * std::cos(t), p * std::sin(t));
    return std::pow(y, n);
  }
};

/**
 * @brief The BinomialRand class <BR>
 * Binomial distribution
 */
template <typename IntType = int>
class RANDLIB_EXPORT BinomialRand : public BinomialDistribution<IntType>
{
public:
  BinomialRand(int number = 1, double probability = 0.5)
  : BinomialDistribution<IntType>(number, probability)
  {
  }

  String Name() const override
  {
    return "Binomial(" + this->toStringWithPrecision(this->GetNumber()) + ", " + this->toStringWithPrecision(this->GetProbability()) + ")";
  }

  using BinomialDistribution<IntType>::SetParameters;
};
} // namespace randlib
