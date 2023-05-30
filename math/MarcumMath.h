#pragma once

#include "math/GammaMath.h"

namespace RandMath
{
/**
 * @fn MarcumP
 * @param mu
 * @param x
 * @param y
 * @param sqrtX √x
 * @param sqrtY √y
 * @param logX log(x)
 * @param logY log(y)
 * @return 1 - Marcum Q-function
 */
double MarcumP(double mu, double x, double y, double sqrtX, double sqrtY, double logX, double logY);

/**
 * @fn MarcumP
 * @param mu
 * @param x
 * @param y
 * @return 1 - Marcum Q-function
 */
double MarcumP(double mu, double x, double y);

/**
 * @fn MarcumQ
 * @param mu
 * @param x
 * @param y
 * @param sqrtX √x
 * @param sqrtY √y
 * @param logX log(x)
 * @param logY log(y)
 * @return Marcum Q-function
 */
double MarcumQ(double mu, double x, double y, double sqrtX, double sqrtY, double logX, double logY);

/**
 * @fn MarcumQ
 * @param mu
 * @param x
 * @param y
 * @return Marcum Q-function
 */
double MarcumQ(double mu, double x, double y);
} // namespace RandMath