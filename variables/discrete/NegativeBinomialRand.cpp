#include "NegativeBinomialRand.h"

template< typename T >
NegativeBinomialRand<T>::NegativeBinomialRand(T number, double probability) :
    G(probability)
{
    setParameters(number, probability);
}

template< typename T >
std::string NegativeBinomialRand<T>::name()
{
    return "Negative Binomial(" + toStringWithPrecision(getNumber()) + ", " + toStringWithPrecision(getProbability()) + ")";
}

template< typename T >
void NegativeBinomialRand<T>::setParameters(T number, double probability)
{
    r = std::max(number, static_cast<T>(1.0));

    p = std::min(std::max(probability, MIN_POSITIVE), 1.0);
    G.setProbability(1 - p);

    pdfCoef = std::pow(1 - p, r) / std::tgamma(r);
    Y.setParameters(r, p / (1 - p));
}

template <>
double NegativeBinomialRand<double>::P(int k) const
{
    return (k < 0) ? 0 : pdfCoef * std::tgamma(r + k) / RandMath::factorial(k) * std::pow(p, k);
}

template <>
double NegativeBinomialRand<int>::P(int k) const
{
    return (k < 0) ? 0 : pdfCoef * RandMath::factorial(r + k) / RandMath::factorial(k) * std::pow(p, k);
}

template< typename T >
double NegativeBinomialRand<T>::F(double x) const
{
    return 1.0 - RandMath::incompleteBetaFun(p, std::floor(x) + 1, r);
}

template<>
double NegativeBinomialRand<double>::variate() const
{
    return variateThroughGammaPoisson();
}

template<>
double NegativeBinomialRand<int>::variate() const
{
    // TODO: find optimal boundary
    if (r < 20)
        return variateThroughGeometric();
    return variateThroughGammaPoisson();
}

template< typename T >
double NegativeBinomialRand<T>::variateThroughGeometric() const
{
    double res = 0;
    for (int i = 0; i != static_cast<int>(r); ++i)
        res += G.variate();
    return res;
}

template< typename T >
double NegativeBinomialRand<T>::variateThroughGammaPoisson() const
{
    return PoissonRand::variate(Y.variate());
}

template class NegativeBinomialRand<int>;
template class NegativeBinomialRand<double>;