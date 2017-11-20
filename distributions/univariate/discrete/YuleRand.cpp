#include "YuleRand.h"

YuleRand::YuleRand(double shape) :
X(shape, 1.0)
{
    SetShape(shape);
}

String YuleRand::Name() const
{
    return "Yule(" + this->toStringWithPrecision(GetShape()) + ")";
}

void YuleRand::SetShape(double shape)
{
    if (shape <= 0.0)
        throw std::invalid_argument("Yule distribution: shape should be positive");
    ro = shape;
    lgamma1pRo = std::lgammal(ro + 1);
    X.SetShape(ro);
}

double YuleRand::P(const int & k) const
{
    return (k < 1) ? 0.0 : std::exp(logP(k));
}

double YuleRand::logP(const int & k) const
{
    if (k < 1)
        return -INFINITY;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + ro + 1);
    y += X.GetLogShape();
    return y;
}

double YuleRand::F(const int & k) const
{
    if (k < 1)
        return 0.0;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + ro + 1);
    y = std::exp(y);
    return 1.0 - k * y;
}

double YuleRand::S(const int & k) const
{
    if (k < 1)
        return 1.0;
    double y = lgamma1pRo;
    y += RandMath::lfact(k - 1);
    y -= std::lgammal(k + ro + 1);
    y = std::exp(y);
    return k * y;
}

int YuleRand::Variate() const
{
    double prob = 1.0 / X.Variate();
    return GeometricRand::Variate(prob, this->localRandGenerator) + 1;
}

int YuleRand::Variate(double shape, RandGenerator &randGenerator)
{
    if (shape <= 0.0)
        return -1;
    double prob = 1.0 / ParetoRand::StandardVariate(shape, randGenerator);
    return GeometricRand::Variate(prob, randGenerator) + 1;
}

void YuleRand::Reseed(unsigned long seed) const
{
    this->localRandGenerator.Reseed(seed);
    X.Reseed(seed + 1);
}

long double YuleRand::Mean() const
{
    return (ro <= 1) ? INFINITY : ro / (ro - 1);
}

long double YuleRand::Variance() const
{
    if (ro <= 2)
        return INFINITY;
    double aux = ro / (ro - 1);
    return aux * aux / (ro - 2);
}

int YuleRand::Mode() const
{
    return 1;
}

long double YuleRand::Skewness() const
{
    if (ro <= 3)
        return INFINITY;
    long double skewness = ro + 1;
    skewness *= skewness;
    skewness *= std::sqrt(ro - 2);
    return skewness / (ro * (ro - 3));
}

long double YuleRand::ExcessKurtosis() const
{
    if (ro <= 4)
        return INFINITY;
    long double numerator = 11 * ro * ro - 49;
    numerator *= ro;
    numerator -= 22;
    long double denominator = ro * (ro - 4) * (ro - 3);
    return ro + 3 + numerator / denominator;
}
