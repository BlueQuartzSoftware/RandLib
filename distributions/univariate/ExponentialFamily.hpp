#pragma once

#include "RandLib.hpp"
#include "RandLib_export.hpp"

#include "math/RandMath.hpp"

namespace randlib
{
template <typename T, typename P>
class RANDLIB_EXPORT ExponentialFamily
{
public:
  ExponentialFamily() = default;

  virtual ~ExponentialFamily() = default;

  //-------------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //-------------------------------------------------------------------------------------------

  virtual P SufficientStatistic(T x) const = 0;

  virtual P SourceParameters() const = 0;

  virtual P SourceToNatural(P sourceParameters) const = 0;

  virtual double LogNormalizer(P theta) const = 0;

  virtual P LogNormalizerGradient(P theta) const = 0;

  virtual double CarrierMeasure(T x) const = 0;

  //-------------------------------------------------------------------------------------------
  // VIRTUAL
  //-------------------------------------------------------------------------------------------

  virtual P NaturalParameters() const
  {
    P sourceParameters = SourceParameters();
    return SourceToNatural(sourceParameters);
  }

  virtual double CrossEntropyAdjusted(P parameters) const
  {
    P theta_p = this->NaturalParameters();
    P theta_q = this->SourceToNatural(parameters);
    double H = this->LogNormalizer(theta_q);
    P grad = this->LogNormalizerGradient(theta_p);
    H -= theta_q * grad;
    return H;
  }

  virtual double EntropyAdjusted() const
  {
    P sourceParameters = this->SourceParameters();
    return CrossEntropyAdjusted(sourceParameters);
  }

  //-------------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //-------------------------------------------------------------------------------------------

  double ProbabilityMeasure(T x) const
  {
    return std::exp(LogProbabilityMeasure(x));
  }

  double LogProbabilityMeasure(T x) const
  {
    P theta = this->NaturalParameters();
    P t = this->SufficientStatistic(x);
    double y = t * theta;
    y -= this->LogNormalizer(theta);
    y += this->CarrierMeasure(x);
    return y;
  }

  double KullbackLeiblerDivergence(P parameters) const
  {
    double KL = this->CrossEntropyAdjusted(parameters);
    KL -= this->EntropyAdjusted();
    return KL;
  }
};
} // namespace randlib
