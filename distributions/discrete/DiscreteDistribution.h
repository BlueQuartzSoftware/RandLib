#ifndef DISCRETE_DISTRIBUTION_H
#define DISCRETE_DISTRIBUTION_H

#include "../ProbabilityDistribution.h"

/**
 *@brief The DiscreteDistribution class
 */
class RANDLIBSHARED_EXPORT DiscreteDistribution : public ProbabilityDistribution
{
public:
    DiscreteDistribution() : ProbabilityDistribution() {}
    virtual ~DiscreteDistribution() {}

    /**
     * @brief P
     * probability to get x
     * @param x
     * @return
     */
    virtual double P(int x) const = 0;

    /**
     * @brief pmf
     * fill vector y by P(x)
     * @param x
     * @param y
     */
    void pmf(const QVector<int> &x, QVector<double> &y) const;

    double Quantile(double probability) const override;

    double Hazard(double x) const override;

    double ExpectedValue(const std::function<double (double)> &funPtr, double startPoint) const override;

    double likelihood(const QVector<int> &sample) const;
    double loglikelihood(const QVector<int> &sample) const;
};

#endif // DISCRETE_DISTRIBUTION_H