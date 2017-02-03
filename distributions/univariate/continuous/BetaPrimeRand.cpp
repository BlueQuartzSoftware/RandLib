#include "BetaPrimeRand.h"

BetaPrimeRand::BetaPrimeRand(double shape1, double shape2)
{
    SetParameters(shape1, shape2);
}

std::string BetaPrimeRand::Name() const
{
    return "Beta Prime(" + toStringWithPrecision(GetAlpha()) + ", " + toStringWithPrecision(GetBeta()) + ")";
}

void BetaPrimeRand::SetParameters(double shape1, double shape2)
{
    B.SetParameters(shape1, shape2);
    alpha = B.GetAlpha();
    beta = B.GetBeta();
}

double BetaPrimeRand::f(double x) const
{
    if (x < 0)
        return 0;
    if (x == 0) {
        if (alpha == 1)
            return GetInverseBetaFunction();
        return (alpha > 1) ? 0.0 : INFINITY;
    }
    double y = (alpha - 1) * std::log(x);
    y -= (alpha + beta) * std::log1p(x);
    return std::exp(y - GetLogBetaFunction());
}

double BetaPrimeRand::F(double x) const
{
    return (x > 0) ? B.F(x / (1.0 + x)) : 0;
}

double BetaPrimeRand::S(double x) const
{
    return (x > 0) ? B.S(x / (1.0 + x)) : 1;
}

double BetaPrimeRand::Variate() const
{
    double x = B.Variate();
    return x / (1.0 - x);
}

void BetaPrimeRand::Sample(std::vector<double> &outputData) const
{
    B.Sample(outputData);
    for (double &var : outputData)
        var = var / (1.0 - var);
}

double BetaPrimeRand::Mean() const
{
    return (beta > 1) ? alpha / (beta - 1) : INFINITY;
}

double BetaPrimeRand::Variance() const
{
    if (beta <= 2)
        return INFINITY;
    double betam1 = beta - 1;
    double numerator = alpha * (alpha + betam1);
    double denominator = (betam1 - 1) * betam1 * betam1;
    return numerator / denominator;
}

double BetaPrimeRand::Median() const
{
    return (alpha == beta) ? 1.0 : quantileImpl(0.5);
}

double BetaPrimeRand::Mode() const
{
    return (alpha < 1) ? 0 : (alpha - 1) / (beta + 1);
}

double BetaPrimeRand::Skewness() const
{
    if (beta <= 3)
        return INFINITY;
    double aux = alpha + beta - 1;
    double skewness = (beta - 2) / (alpha * aux);
    skewness = std::sqrt(skewness);
    aux += alpha;
    aux += aux;
    return aux * skewness / (beta - 3);
}

double BetaPrimeRand::ExcessKurtosis() const
{
    if (beta <= 4)
        return INFINITY;
    double betam1 = beta - 1;
    double numerator = betam1 * betam1 * (beta - 2) / (alpha * (alpha + betam1));
    numerator += 5 * beta - 11;
    double denominator = (beta - 3) * (beta - 4);
    return 6 * numerator / denominator;
}

double BetaPrimeRand::quantileImpl(double p) const
{
    double x = B.Quantile(p);
    return x / (1.0 - x);
}

double BetaPrimeRand::quantileImpl1m(double p) const
{
    double x = B.Quantile1m(p);
    return x / (1.0 - x);
}

std::complex<double> BetaPrimeRand::CFImpl(double t) const
{
    /// if no singularity - simple numeric integration
    if (alpha > 1)
        return UnivariateProbabilityDistribution::CFImpl(t);

    double re = RandMath::integral([this, t] (double x) {
        if (x <= 0)
            return 0.0;
        double y = std::pow(1 + x, -alpha - beta) - 1.0;
        y *= std::pow(x, alpha - 1);
        return y;
    }, 0, 1);

    re += 1.0 / alpha;
    re *= GetInverseBetaFunction();

    re += RandMath::integral([this, t] (double x)
    {
        if (x >= 1.0)
            return 0.0;
        double denom = 1.0 - x;
        double p = 1.0 + x / denom;
        double y = std::cos(p * t) * f(p);
        denom *= denom;
        return y / denom;
    },
    0.0, 1.0);

    double mode = Mode();
    double im = ExpectedValue([this, t] (double x)
    {
        return std::sin(t * x);
    }, 0.0, mode);
    im += ExpectedValue([this, t] (double x)
    {
        return std::sin(t * x);
    }, mode, INFINITY);
    return std::complex<double>(re, im);
}
