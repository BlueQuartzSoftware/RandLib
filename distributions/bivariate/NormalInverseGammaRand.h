#ifndef NORMALINVERSEGAMMARAND_H
#define NORMALINVERSEGAMMARAND_H

#include "BivariateProbabilityDistribution.h"
#include "../univariate/continuous/StudentTRand.h"
#include "../univariate/continuous/InverseGammaRand.h"

/**
 * @brief The NormalInverseGammaRand class
 * Normal-inverse-gamma distribution
 *
 * Notation: X ~ NIG(μ, λ, α, β)
 */
class RANDLIBSHARED_EXPORT NormalInverseGammaRand : public BivariateProbabilityDistribution<StudentTRand, InverseGammaRand>
{
    double mu, lambda;
    double alpha, beta;
    double pdfCoef;

public:
    NormalInverseGammaRand(double location = 0, double precision = 1, double shape = 1, double rate = 1);
    std::string Name() const override;
    DoublePair MinValue() const { return DoublePair(-INFINITY, 0); }
    DoublePair MaxValue() const { return DoublePair(INFINITY, INFINITY); }

    void SetParameters(double location, double precision, double shape, double rate);
    inline double GetLocation() const { return mu; }
    inline double GetPrecision() const { return lambda; }
    inline double GetShape() const { return alpha; }
    inline double GetRate() const { return beta; }

    double f(const DoublePair &point) const override;
    double logf(const DoublePair &point) const override;
    double F(const DoublePair & point) const override;
    DoublePair Variate() const override;

    double Correlation() const override;
};

#endif // NORMALINVERSEGAMMARAND_H
