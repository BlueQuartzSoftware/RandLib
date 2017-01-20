#ifndef HESTONPROCESS_H
#define HESTONPROCESS_H

#include "CoxIngersollRossProcess.h"
#include "../distributions/bivariate/BivariateNormalRand.h"

/**
 * @brief The HestonProcess class
 * dX(t) = μX(t)dt + (V(t))^(1/2)X(t)dB(t)),
 * where V(t) is CIR process (volatility) satisfying:
 * dV(t) = β(α - V(t))dt + σ(V(t))^(1/2)dW(t).
 * B(t) and W(t) are correlated Wiener processes with dB(t)dW(t) = ρdt,
 * μ is the drift term (the rate of return of the asSet),
 * α is the mean of V(t),
 * β is the reversion speed of V(t),
 * σ is the volatility of V(t),
 * ρ is the correlation between B(t) and W(t).
 *
 * Notation: SV(t | μ, α, β, σ, X0, V0, ρ)
 *
 * No closed-form solution.
 *
 * @todo Add exact simulation and use instead of Euler discretization
 */
class RANDLIBSHARED_EXPORT HestonProcess : public StochasticProcess<double>
{
    double mu, rho;
    CoxIngersollRossProcess V;
    BivariateNormalRand BW;

public:
    HestonProcess(double drift, double volatilityMean, double reversionSpeed, double volatility, double initialValue, double volatilityInitialValue, double correlation, double deltaT = 1.0);

    inline double GetDrift() { return mu; }
    inline double GetVolatilityMean() { return V.GetMean(); }
    inline double GetReversionSpeed() { return V.GetReversionSpeed(); }
    inline double GetVolatility() { return V.GetVolatility(); }
    inline double GetCorrelation() { return rho; }

private:
    void NextImpl() override;

    double MeanImpl(double t) const override;
    double VarianceImpl(double t) const override;
};

#endif // HESTONPROCESS_H
