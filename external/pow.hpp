/*################################################################################
  ##
  ##   Copyright (C) 2016-2023 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * compile-time power function
 */

#pragma once

#include "gcem_options.hpp"
#include "log.hpp"

#include <type_traits>

namespace internal
{
using namespace gcem;

template <typename T1, typename T2>
constexpr T1 pow_integral(const T1 base, const T2 exp_term) noexcept;

// see https://en.wikipedia.org/wiki/Euler%27s_continued_fraction_formula

#if __cplusplus >= 201402L // C++14 version

template <typename T>
constexpr T exp_cf_recur(const T x, const int depth_end) noexcept
{
  int depth = GCEM_EXP_MAX_ITER_SMALL - 1;
  T res = T(1);

  while(depth > depth_end - 1)
  {
    res = T(1) + x / T(depth - 1) - x / depth / res;

    --depth;
  }

  return res;
}

#else // C++11 version

template <typename T>
constexpr T exp_cf_recur(const T x, const int depth) noexcept
{
  return (depth < GCEM_EXP_MAX_ITER_SMALL ? // if
              T(1) + x / T(depth - 1) - x / depth / exp_cf_recur(x, depth + 1) :
              // else
              T(1));
}

#endif

template <typename T>
constexpr T exp_cf(const T x) noexcept
{
  return (T(1) / (T(1) - x / exp_cf_recur(x, 2)));
}

template <typename T>
constexpr T exp_split(const T x) noexcept
{
  return (static_cast<T>(internal::pow_integral(GCEM_E, checks::find_whole(x))) * exp_cf(checks::find_fraction(x)));
}

template <typename T>
constexpr T exp_check(const T x) noexcept
{
  return (helpers::is_nan(x) ? GCLIM<T>::quiet_NaN() :
                               //
              helpers::is_neginf(x) ? T(0) :
                                      // indistinguishable from zero
              GCLIM<T>::min() > helpers::abs(x) ? T(1) :
                                                  //
              helpers::is_posinf(x) ? GCLIM<T>::infinity() :
                                      //
              helpers::abs(x) < T(2) ? exp_cf(x) :
                                       exp_split(x));
}

/**
 * Compile-time exponential function
 *
 * @param x a real-valued input.
 * @return \f$ \exp(x) \f$ using \f[ \exp(x) = \dfrac{1}{1-\dfrac{x}{1+x-\dfrac{\frac{1}{2}x}{1 + \frac{1}{2}x - \dfrac{\frac{1}{3}x}{1 + \frac{1}{3}x - \ddots}}}} \f]
 * The continued fraction argument is split into two parts: \f$ x = n + r \f$, where \f$ n \f$ is an integer and \f$ r \in [-0.5,0.5] \f$.
 */
template <typename T>
constexpr return_t<T> exp(const T x) noexcept
{
  return internal::exp_check(static_cast<return_t<T>>(x));
}

template <typename T1, typename T2>
constexpr T1 pow_integral_compute(const T1 base, const T2 exp_term) noexcept;

// integral-valued powers using method described in
// https://en.wikipedia.org/wiki/Exponentiation_by_squaring

template <typename T1, typename T2>
constexpr T1 pow_integral_compute_recur(const T1 base, const T1 val, const T2 exp_term) noexcept
{
  return (exp_term > T2(1) ? (helpers::is_odd(exp_term) ? pow_integral_compute_recur(base * base, val * base, exp_term / 2) : pow_integral_compute_recur(base * base, val, exp_term / 2)) :
                             (exp_term == T2(1) ? val * base : val));
}

template <typename T1, typename T2, typename std::enable_if<std::is_signed<T2>::value>::type* = nullptr>
constexpr T1 pow_integral_sgn_check(const T1 base, const T2 exp_term) noexcept
{
  return (exp_term < T2(0) ? //
              T1(1) / pow_integral_compute(base, -exp_term) :
              //
              pow_integral_compute_recur(base, T1(1), exp_term));
}

template <typename T1, typename T2, typename std::enable_if<!std::is_signed<T2>::value>::type* = nullptr>
constexpr T1 pow_integral_sgn_check(const T1 base, const T2 exp_term) noexcept
{
  return (pow_integral_compute_recur(base, T1(1), exp_term));
}

template <typename T1, typename T2>
constexpr T1 pow_integral_compute(const T1 base, const T2 exp_term) noexcept
{
  return (exp_term == T2(3) ? base * base * base :
          exp_term == T2(2) ? base * base :
          exp_term == T2(1) ? base :
          exp_term == T2(0) ? T1(1) :
                              // check for overflow
              exp_term == GCLIM<T2>::min() ? T1(0) :
          exp_term == GCLIM<T2>::max()     ? GCLIM<T1>::infinity() :
                                             // else
                                             pow_integral_sgn_check(base, exp_term));
}

template <typename T1, typename T2, typename std::enable_if<std::is_integral<T2>::value>::type* = nullptr>
constexpr T1 pow_integral_type_check(const T1 base, const T2 exp_term) noexcept
{
  return pow_integral_compute(base, exp_term);
}

template <typename T1, typename T2, typename std::enable_if<!std::is_integral<T2>::value>::type* = nullptr>
constexpr T1 pow_integral_type_check(const T1 base, const T2 exp_term) noexcept
{
  // return GCLIM<return_t<T1>>::quiet_NaN();
  return pow_integral_compute(base, static_cast<llint_t>(exp_term));
}

template <typename T1, typename T2>
constexpr T1 pow_integral(const T1 base, const T2 exp_term) noexcept
{
  return pow_integral_type_check(base, exp_term);
}

template <typename T>
constexpr T pow_dbl(const T base, const T exp_term) noexcept
{
  return internal::exp(exp_term * nonstd::log(base));
}

template <typename T1, typename T2, typename TC = common_t<T1, T2>, typename std::enable_if<!std::is_integral<T2>::value>::type* = nullptr>
constexpr TC pow_check(const T1 base, const T2 exp_term) noexcept
{
  return (base < T1(0) ? GCLIM<TC>::quiet_NaN() :
                         //
                         pow_dbl(static_cast<TC>(base), static_cast<TC>(exp_term)));
}

template <typename T1, typename T2, typename TC = common_t<T1, T2>, typename std::enable_if<std::is_integral<T2>::value>::type* = nullptr>
constexpr TC pow_check(const T1 base, const T2 exp_term) noexcept
{
  return pow_integral(base, exp_term);
}
} // namespace internal

namespace nonstd
{
/**
 * Compile-time power function
 *
 * @param base a real-valued input.
 * @param exp_term a real-valued input.
 * @return Computes \c base raised to the power \c exp_term. In the case where \c exp_term is integral-valued, recursion by squaring is used, otherwise \f$ \text{base}^{\text{exp\_term}} =
 * e^{\text{exp\_term} \log(\text{base})} \f$
 */

template <typename T1, typename T2>
constexpr gcem::common_t<T1, T2> pow(const T1 base, const T2 exp_term) noexcept
{
  return internal::pow_check(base, exp_term);
}
} // namespace nonstd
