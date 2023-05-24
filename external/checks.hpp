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

#pragma once

#include "floor.hpp"
#include "gcem_options.hpp"
#include "helpers.hpp"

namespace checks
{
using namespace gcem;
template<typename T>
constexpr
llint_t
find_exponent(const T x, const llint_t exponent)
noexcept
{
    return( // < 1
            x < T(1e-03)  ? \
                find_exponent(x * T(1e+04), exponent - llint_t(4)) :
            x < T(1e-01)  ? \
                find_exponent(x * T(1e+02), exponent - llint_t(2)) :
            x < T(1)  ? \
                find_exponent(x * T(10), exponent - llint_t(1)) :
            // > 10
            x > T(10) ? \
                find_exponent(x / T(10), exponent + llint_t(1)) :
            x > T(1e+02) ? \
                find_exponent(x / T(1e+02), exponent + llint_t(2)) :
            x > T(1e+04) ? \
                find_exponent(x / T(1e+04), exponent + llint_t(4)) :
            // else
                exponent );
}

template<typename T>
constexpr
llint_t
find_whole(const T x)
noexcept
{
    return( helpers::abs(x - internal::floor_check(x)) >= T(0.5) ? \
            // if 
                static_cast<llint_t>(internal::floor_check(x) + nonstd::sgn(x)) :
            // else 
                static_cast<llint_t>(internal::floor_check(x)) );
}

template<typename T>
constexpr
T
find_fraction(const T x)
noexcept
{
    return( helpers::abs(x - internal::floor_check(x)) >= T(0.5) ? \
            // if 
                x - internal::floor_check(x) - nonstd::sgn(x) : 
            //else 
                x - internal::floor_check(x) );
}
}
