#pragma once

#include "math/GammaMath.hpp"

#include <complex>

namespace randlib::RandMath
{
/**
 * @fn logBeta
 * Calculate logarithm of beta function
 * @param a positive parameter
 * @param b positive parameter
 * @return log(B(a, b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a + b))
 */
long double logBeta(long double a, long double b);

/**
 * @fn ibeta
 * Fast calculation of regularized beta function, using precalculated values
 * @param x
 * @param a
 * @param b
 * @param logBetaFun log(B(a, b))
 * @param logX log(x)
 * @param log1mX log(1-x)
 * @return I(x, a, b) = B(x, a, b) / B(a, b)
 */
double ibeta(double x, double a, double b, double logBetaFun, double logX, double log1mX);

/**
 * @fn ibeta
 * Calculate regularized beta function
 * @param x
 * @param a
 * @param b
 * @return I(x, a, b) = B(x, a, b) / B(a, b)
 */
double ibeta(double x, double a, double b);
} // namespace randlib::RandMath
