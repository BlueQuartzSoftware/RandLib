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

    p = probability;
    if (p >= 1.0 || p < 0.0)
        p = 0.5;
    q = 1.0 - p;

    /// use Y OR G (depends on p and r)
    G.setProbability(q);

    pdfCoef = std::pow(q, r) / std::tgamma(r);
    Y.setParameters(r, p / q);
}

template <>
double NegativeBinomialRand<double>::P(int k) const
{
    return (k < 0) ? 0 : pdfCoef * std::tgamma(r + k) / RandMath::factorial(k) * std::pow(p, k);
}

template <>
double NegativeBinomialRand<int>::P(int k) const
{
    return (k < 0) ? 0 : pdfCoef * RandMath::factorial(r + k - 1) / RandMath::factorial(k) * std::pow(p, k);
}

template< typename T >
double NegativeBinomialRand<T>::F(double x) const
{
    if (x < 0.0)
        return 0.0;
    return 1.0 - RandMath::regularizedBetaFun(p, std::floor(x) + 1, r);
}

template<>
double NegativeBinomialRand<double>::variate() const
{
    return variateThroughGammaPoisson();
}

template<>
double NegativeBinomialRand<int>::variate() const
{
    if (r < 10) /// also consider p and do sample function!
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

template< typename T >
double NegativeBinomialRand<T>::Mean() const
{
    return p * r / q;
}

template< typename T >
double NegativeBinomialRand<T>::Variance() const
{
    return p * r / (q * q);
}

template< typename T >
std::complex<double> NegativeBinomialRand<T>::CF(double t) const
{
    std::complex<double> denominator(0, t);
    denominator = 1.0 - p * std::exp(denominator);
    return std::pow(q / denominator, r);
}

template< typename T >
double NegativeBinomialRand<T>::Mode() const
{
    return (r > 1) ? std::floor((r - 1) * p / q) : 0;
}

template< typename T >
double NegativeBinomialRand<T>::Skewness() const
{
    return (1 + p) / std::sqrt(p * r);
}

template< typename T >
double NegativeBinomialRand<T>::ExcessKurtosis() const
{
    double kurtosis = q * q;
    kurtosis /= p;
    kurtosis += 6;
    return kurtosis / r;
}

template class NegativeBinomialRand<int>;
template class NegativeBinomialRand<double>;