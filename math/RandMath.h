#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "math/NumericMath.h"

#include "RandLib_export.h"

namespace randlib::RandMath
{
/**
 * @fn sign
 * @param x
 * @return sign of x
 */
int sign(double x);

/**
 * @fn atan
 * @param x
 * @return safe atan(x)
 */
double atan(double x);

/**
 * @fn log1pexp
 * @param x
 * @return log(1 + exp(x))
 */
double softplus(double x);

/**
 * @fn log1mexp
 * @param x
 * @return log(1 - exp(x))
 */
double log1mexp(double x);

/**
 * @fn logexpm1
 * @param x
 * @return log(exp(x) - 1)
 */
double logexpm1l(double x);

/**
 * @fn log2mexp
 * @param x
 * @return log(2 - exp(x))
 */
double log2mexp(double x);

/**
 * @fn erfinv
 * @param p
 * @return inverse error function: such x that erf(x) = p
 */
double erfinv(double p);

/**
 * @fn erfcinv
 * @param p
 * @return inverse complementary error function: such x that erfc(x) = p
 */
double erfcinv(double p);

/**
 * @fn logBesselI
 * Calculates logarithm of modified Bessel function of the 1st kind
 * @param nu
 * @param x
 * @return log(I_ν(x))
 */
long double logBesselI(double nu, double x);

/**
 * @fn logBesselK
 * Calculates logarithm of modified Bessel function of the 2nd kind
 * @param nu
 * @param x
 * @return log(K_ν(x))
 */
long double logBesselK(double nu, double x);

/**
 * @fn W0Lambert
 * @param x
 * @param epsilon
 * @return W0 branch of Lambert W function
 */
double W0Lambert(double x, double epsilon = 1e-11);

/**
 * @fn W1Lambert
 * @param x
 * @param epsilon
 * @return W-1 branch of Lambert W function
 */
double Wm1Lambert(double x, double epsilon = 1e-11);
} // namespace randlib::RandMath
