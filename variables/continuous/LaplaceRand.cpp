#include "LaplaceRand.h"

LaplaceRand::LaplaceRand(double location, double scale)
{
    setLocation(location);
    setScale(scale);
}

void LaplaceRand::setName()
{
    nameStr = "Laplace(" + toStringWithPrecision(getLocation()) + ", " + toStringWithPrecision(getScale()) + ")";
}

void LaplaceRand::setLocation(double location)
{
    mu = location;
    setName();
}

void LaplaceRand::setScale(double scale)
{
    b = std::max(scale, MIN_POSITIVE);
    bInv = 1.0 / b;
    setName();
}

double LaplaceRand::f(double x) const
{
    double y = -std::fabs(x - mu);
    y *= bInv;
    y = std::exp(y);
    y *= bInv;
    return .5 * y;
}

double LaplaceRand::F(double x) const
{
    double y = x - mu;
    y *= bInv;
    if (x < mu)
        return .5 * std::exp(y);
    y = -.5 * std::exp(-y);
    return y + 1;
}

double LaplaceRand::variate() const
{
    double e = ExponentialRand::variate(bInv);
    return mu + (((signed)RandGenerator::variate() > 0) ? e : -e);
}

bool LaplaceRand::fitToData(const QVector<double> &sample)
{
    if (sample.size() == 0)
        return false;

    /// Calculate median
    /// we use root-finding algorithm for median search
    /// but note, that it is better to use median-for-median algorithm
    double median = 0.0;
    double min = sample[0], max = min;
    for (double var : sample) {
        min = std::min(var, min);
        max = std::max(var, max);
    }

    if (!RandMath::findRoot([sample] (double med)
    {
        /// sum of sign(x) - derivative of sum of abs(x)
        double x = 0;
        for (double var : sample) {
            if (var > med)
                ++x;
            else if (var < med)
                --x;
        }
        return x;
    },
    min, max, median
    ))
        return false;


    /// Calculate scale
    double deviation = 0.0;
    for (double var : sample) {
        deviation += std::fabs(var - median);
    }
    deviation /= sample.size();

    setLocation(median);
    setScale(deviation);
    return true;
}

