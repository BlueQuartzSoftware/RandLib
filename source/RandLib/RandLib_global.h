#pragma once

#include <climits>
#include <cmath>
#include <string>

#if !defined(INFINITY) || !defined(NAN)
#include <limits>
#endif

namespace RandLib
{
inline constexpr long double MIN_POSITIVE = 1e-21l;

template <typename T>
using Pair = std::pair<T, T>;

template <typename T>
Pair<T> operator+(const Pair<T>& l, const Pair<T>& r)
{
  return {l.first + r.first, l.second + r.second};
}

template <typename T>
Pair<T> operator-(const Pair<T>& l, const Pair<T>& r)
{
  return {l.first - r.first, l.second - r.second};
}

template <typename T>
T operator*(const Pair<T>& l, const Pair<T>& r)
{
  return l.first * r.first + l.second * r.second;
}

template <typename T>
using Triplet = std::tuple<T, T, T>;

using IntPair = Pair<int>;
using DoublePair = Pair<double>;
using LongDoublePair = Pair<long double>;

using IntTriplet = Triplet<int>;
using DoubleTriplet = Triplet<double>;
using LongDoubleTriplet = Triplet<long double>;

#ifndef INFINITY
inline constexpr long double INFINITY = std::numeric_limits<long double>::infinity();
#endif

#ifndef NAN
inline constexpr long double NAN = std::numeric_limits<long double>::quiet_NaN();
#endif

#ifndef M_E
inline constexpr long double M_E = 2.71828182845904523536l;
#endif

#ifndef M_LOG2E
inline constexpr long double M_LOG2E = 1.44269504088896340760l;
#endif

#ifndef M_LOG10E
inline constexpr long double M_LOG10E = 0.43429448190325182765l;
#endif

#ifndef M_LN2
inline constexpr long double M_LN2 = 0.69314718055994530942l;
#endif

#ifndef M_LN10
inline constexpr long double M_LN10 = 2.30258509299404568402l;
#endif

#ifndef M_PI
inline constexpr long double M_PI = 3.14159265358979323846l;
#endif

#ifndef M_PI_2
inline constexpr long double M_PI_2 = 1.57079632679489661923l;
#endif

#ifndef M_1_PI
inline constexpr long double M_1_PI = 0.31830988618379067154l;
#endif

#ifndef M_2_PI
inline constexpr long double M_2_PI = 0.63661977236758134308l;
#endif

#ifndef M_2_SQRTPI
inline constexpr long double M_2_SQRTPI = 1.12837916709551257390l;
#endif

#ifndef M_SQRT2
inline constexpr long double M_SQRT2 = 1.41421356237309504880l;
#endif

#ifndef M_SQRT1_2
inline constexpr long double M_SQRT1_2 = 0.70710678118654752440l;
#endif

#ifndef M_SQRT5
inline constexpr long double M_SQRT5 = 2.23606797749978969640l;
#endif
} // namespace RandLib
