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

namespace RandMath
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

/**
 * @fn findMin
 * Combined Brent's method
 * @param funPtr
 * @param abc lower boundary / middle / upper boundary
 * @param fx funPtr(b)
 * @param root such x that funPtr(x) is min
 * @param epsilon tolerance
 * @return true if success
 */
template <typename RealType>
bool findMin(const std::function<double(RealType)>& funPtr, const Triplet<RealType>& abc, double& fx, RealType& root, double epsilon)
{
  static constexpr double K = 0.5 * (3 - M_SQRT5);
  auto [a, x, c] = abc;
  double w = x, v = x, fw = fx, fv = fx;
  double d = c - a, e = d;
  double u = a - 1;
  do
  {
    double g = e;
    e = d;
    bool acceptParabolicU = false;
    if(x != w && x != v && w != v && fx != fw && fx != fv && fw != fv)
    {
      if(v < w)
      {
        if(x < v)
          u = parabolicMinimum<RealType>(x, v, w, fx, fv, fw);
        else if(x < w)
          u = parabolicMinimum<RealType>(v, x, w, fv, fx, fw);
        else
          u = parabolicMinimum<RealType>(v, w, x, fv, fw, fx);
      }
      else
      {
        if(x < w)
          u = parabolicMinimum<RealType>(x, w, v, fx, fv, fw);
        else if(x < v)
          u = parabolicMinimum<RealType>(w, x, v, fw, fx, fv);
        else
          u = parabolicMinimum<RealType>(w, v, x, fw, fv, fx);
      }
      double absumx = std::fabs(u - x);
      if(u >= a + epsilon && u <= c - epsilon && absumx < 0.5 * g)
      {
        acceptParabolicU = true; /// accept u
        d = absumx;
      }
    }

    if(!acceptParabolicU)
    {
      /// use golden ratio instead of parabolic approximation
      if(x < 0.5 * (c + a))
      {
        d = c - x;
        u = x + K * d; /// golden ratio [x, c]
      }
      else
      {
        d = x - a;
        u = x - K * d; /// golden ratio [a, x]
      }
    }

    if(std::fabs(u - x) < epsilon)
    {
      u = x + epsilon * sign(u - x); /// setting the closest distance between u and x
    }

    double fu = funPtr(u);
    if(fu <= fx)
    {
      if(u >= x)
        a = x;
      else
        c = x;
      v = w;
      w = x;
      x = u;
      fv = fw;
      fw = fx;
      fx = fu;
    }
    else
    {
      if(u >= x)
        c = u;
      else
        a = u;
      if(fu <= fw || w == x)
      {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      }
      else if(fu <= fv || v == x || v == w)
      {
        v = u;
        fv = fu;
      }
    }
  } while(0.49 * (c - a) > epsilon);
  root = x;
  return true;
}

/**
 * @fn findMin
 * Combined Brent's method
 * @param funPtr
 * @param closePoint point that is nearby minimum
 * @param root such x that funPtr(x) is min
 * @param epsilon tolerance
 * @return true if success
 */
template <typename RealType>
bool findMin(const std::function<double(RealType)>& funPtr, RealType closePoint, RealType& root, long double epsilon = 1e-8)
{
  Triplet<RealType> abc;
  static constexpr double K = 0.5 * (M_SQRT5 + 1);
  static constexpr int L = 100;
  double a = closePoint, fa = funPtr(a);
  double b = a + 1.0, fb = funPtr(b);
  double c, fc;
  if(fb < fa)
  {
    c = b + K * (b - a);
    fc = funPtr(c);
    /// we go to the right
    while(fc < fb)
    {
      /// parabolic interpolation
      double u = parabolicMinimum<RealType>(a, b, c, fa, fb, fc);
      double cmb = c - b;
      double fu, uLim = c + L * cmb;
      if(u < c && u > b)
      {
        fu = funPtr(u);
        if(fu < fc)
        {
          abc = std::make_tuple(b, u, c);
          return findMin(funPtr, abc, fu, root, epsilon);
        }
        if(fu > fb)
        {
          abc = std::make_tuple(a, b, u);
          return findMin(funPtr, abc, fb, root, epsilon);
        }
        u = c + K * cmb;
        fu = funPtr(u);
      }
      else if(u > c && u < uLim)
      {
        fu = funPtr(u);
        if(fu < fc)
        {
          b = c;
          c = u;
          u = c + K * cmb;
          fb = fc, fc = fu, fu = funPtr(u);
        }
      }
      else if(u > uLim)
      {
        u = uLim;
        fu = funPtr(u);
      }
      else
      {
        u = c + K * cmb;
        fu = funPtr(u);
      }
      a = b;
      b = c;
      c = u;
      fa = fb;
      fb = fc;
      fc = fu;
    }
    abc = std::make_tuple(a, b, c);
    return findMin(funPtr, abc, fb, root, epsilon);
  }
  else
  {
    c = b;
    fc = fb;
    b = a;
    fb = fa;
    a = b - K * (c - b);
    fa = funPtr(a);
    /// go to the left
    while(fa < fb)
    {
      /// parabolic interpolation
      double u = parabolicMinimum<RealType>(a, b, c, fa, fb, fc);
      double bma = b - a;
      double fu, uLim = a - L * bma;
      if(u < b && u > a)
      {
        fu = funPtr(u);
        if(fu < fa)
        {
          abc = std::make_tuple(a, u, b);
          return findMin(funPtr, abc, fu, root, epsilon);
        }
        if(fu > fb)
        {
          abc = std::make_tuple(u, b, c);
          return findMin(funPtr, abc, fb, root, epsilon);
        }
        u = a - K * bma;
        fu = funPtr(u);
      }
      else if(u < a && u > uLim)
      {
        fu = funPtr(u);
        if(fu < fa)
        {
          b = a;
          a = u;
          u = a - K * bma;
          fb = fa, fa = fu, fu = funPtr(u);
        }
      }
      else if(u < uLim)
      {
        u = uLim;
        fu = funPtr(u);
      }
      else
      {
        u = a - K * bma;
        fu = funPtr(u);
      }
      c = b;
      b = a;
      a = u;
      fc = fb;
      fb = fa;
      fa = fu;
    }
    abc = std::make_tuple(a, b, c);
    return findMin(funPtr, abc, fb, root, epsilon);
  }
}
} // namespace RandMath
