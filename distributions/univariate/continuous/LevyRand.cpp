#include "LevyRand.h"
#include "NormalRand.h"

LevyRand::LevyRand(double location, double scale)
    : StableRand(0.5, 1, scale, location)
{
}

std::string LevyRand::name()
{
    return "Levy(" + toStringWithPrecision(getLocation()) + ", " + toStringWithPrecision(getScale()) + ")";
}

double LevyRand::f(double x) const
{
    return StableRand::pdfLevy(x);
}

double LevyRand::F(double x) const
{
    return StableRand::cdfLevy(x);
}

double LevyRand::variate() const
{
    double rv = NormalRand::standardVariate();
    rv *= rv;
    rv = sigma / rv;
    return mu + rv;
}

double LevyRand::variate(double location, double scale)
{
    return location + scale * standardVariate();
}

double LevyRand::standardVariate()
{
    double rv = NormalRand::standardVariate();
    return 1.0 / (rv * rv);
}

std::complex<double> LevyRand::CF(double t) const
{
    std::complex<double> y(0.0, -2 * sigma * t);
    y = -std::sqrt(y);
    y += std::complex<double>(0.0, mu * t);
    return std::exp(y);
}

double LevyRand::Quantile(double p) const
{
    if (p == 0.0)
        return mu;
    double x = pdfCoef * M_SQRT2PI / NormalRand::standardQuantile(0.5 * p);
    return mu + x * x;
}

double LevyRand::Mode() const
{
    return sigma / 3.0 + mu;
}

bool LevyRand::checkValidity(const std::vector<double> &sample)
{
    for (double var : sample) {
        if (var <= mu)
            return false;
    }
    return true;
}

bool LevyRand::fitScaleMLE(const std::vector<double> &sample)
{
    double n = sample.size();
    if (n <= 0 || !checkValidity(sample))
        return false;
    long double invSum = 0.0;
    for (double var : sample)
        invSum += 1.0 / (var - mu);
    setScale(n / invSum);
    return true;
}
