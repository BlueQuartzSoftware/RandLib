#include "RandMath.h"

#include <functional>

using namespace randlib;
using namespace randlib::RandMath;

namespace
{
double erfinvChebyshevSeries(double x, long double t, const long double* array, int size)
{
  /// We approximate inverse erf via Chebyshev polynomials
  long double Tn = t, Tnm1 = 1, sum = 0.0;
  for(int i = 1; i != size; ++i)
  {
    sum += array[i] * Tn;
    /// calculate next Chebyshev polynomial
    double temp = Tn;
    Tn *= 2 * t;
    Tn -= Tnm1;
    Tnm1 = temp;
  }
  return x * (array[0] + sum);
}

double erfinvAux1(double beta)
{
  /// |1 - p| < 5e-16
  static constexpr long double D5 = -9.199992358830151031278420l, D6 = 2.794990820124599493768426l;
  static constexpr int muSize = 26;
  static constexpr long double mu[muSize] = {.9885750640661893136460358l,
                                             .0108577051845994776160281l,
                                             -.0017511651027627952594825l,
                                             .0000211969932065633437984l,
                                             .0000156648714042435087911l,
                                             -.05190416869103124261e-5l,
                                             -.00371357897426717780e-5l,
                                             .00012174308662357429e-5l,
                                             -.00001768115526613442e-5l,
                                             -.119372182556161e-10l,
                                             .003802505358299e-10l,
                                             -.000660188322362e-10l,
                                             -.000087917055170e-10l,
                                             -.3506869329e-15l,
                                             -.0697221497e-15l,
                                             -.0109567941e-15l,
                                             -.0011536390e-15l,
                                             -.0000263938e-15l,
                                             .05341e-20l,
                                             -.22610e-20l,
                                             .09552e-20l,
                                             -.05250e-20l,
                                             .02487e-20l,
                                             -.01134e-20l,
                                             .00420e-20l};
  return erfinvChebyshevSeries(beta, D5 / std::sqrt(beta) + D6, mu, muSize);
}

double erfinvAux2(double beta)
{
  /// 5e-16 < |1 - p| < 0.025
  static constexpr long double D3 = -0.5594576313298323225436913l, D4 = 2.287915716263357638965891l;
  static constexpr int deltaSize = 38;
  static constexpr long double delta[deltaSize] = {.9566797090204925274526373l,
                                                   -.0231070043090649036999908l,
                                                   -.0043742360975084077333218l,
                                                   -.0005765034226511854809364l,
                                                   -.0000109610223070923931242l,
                                                   .0000251085470246442787982l,
                                                   .0000105623360679477511955l,
                                                   .27544123300306391503e-5l,
                                                   .04324844983283380689e-5l,
                                                   -.00205303366552086916e-5l,
                                                   -.00438915366654316784e-5l,
                                                   -.00176840095080881795e-5l,
                                                   -.00039912890280463420e-5l,
                                                   -.00001869324124559212e-5l,
                                                   .00002729227396746077e-5l,
                                                   .00001328172131565497e-5l,
                                                   .318342484482286e-10l,
                                                   .016700607751926e-10l,
                                                   -.020364649611537e-10l,
                                                   -.009648468127965e-10l,
                                                   -.002195672778128e-10l,
                                                   -.000095689813014e-10l,
                                                   .000137032572230e-10l,
                                                   .000062538505417e-10l,
                                                   .000014584615266e-10l,
                                                   .1078123993e-15l,
                                                   -.0709229988e-15l,
                                                   -.0391411775e-15l,
                                                   -.0111659209e-15l,
                                                   -.0015770366e-15l,
                                                   .0002853149e-15l,
                                                   .0002716662e-15l,
                                                   .0000176835e-15l,
                                                   .09828e-20l,
                                                   .20464e-20l,
                                                   .08020e-20l,
                                                   .01650e-20l};
  return erfinvChebyshevSeries(beta, D3 * beta + D4, delta, deltaSize);
}

double erfinvAux3(double beta)
{
  /// 0.8 < p < 0.975
  static constexpr long double D1 = -1.548813042373261659512742l, D2 = 2.565490123147816151928163l;
  static constexpr int lambdaSize = 27;
  static constexpr long double lambda[lambdaSize] = {.9121588034175537733059200l,
                                                     -.0162662818676636958546661l,
                                                     .0004335564729494453650589l,
                                                     .0002144385700744592065205l,
                                                     .26257510757648130176e-5l,
                                                     -.30210910501037969912e-5l,
                                                     -.00124060618367572157e-5l,
                                                     .00624066092999917380e-5l,
                                                     -.00005401247900957858e-5l,
                                                     -.00014232078975315910e-5l,
                                                     .343840281955305e-10l,
                                                     .335848703900138e-10l,
                                                     -.014584288516512e-10l,
                                                     -.008102174258833e-10l,
                                                     .000525324085874e-10l,
                                                     .000197115408612e-10l,
                                                     -.000017494333828e-10l,
                                                     -.4800596619e-15l,
                                                     .0557302987e-15l,
                                                     .0116326054e-15l,
                                                     -.0017262489e-15l,
                                                     -.0002784973e-15l,
                                                     .0000524481e-15l,
                                                     .65270e-20l,
                                                     -.15707e-20l,
                                                     -.01475e-20l,
                                                     .00450e-20l};
  return erfinvChebyshevSeries(beta, D1 * beta + D2, lambda, lambdaSize);
}

double erfinvAux4(double p)
{
  /// 0 < p < 0.8
  static constexpr int xiSize = 39;
  static constexpr long double xi[xiSize] = {.9928853766189408231495800l,
                                             .1204675161431044864647846l,
                                             .0160781993420999447267039l,
                                             .0026867044371623158279591l,
                                             .0004996347302357262947170l,
                                             .0000988982185991204409911l,
                                             .0000203918127639944337340l,
                                             .43272716177354218758e-5l,
                                             .09380814128593406758e-5l,
                                             .02067347208683427411e-5l,
                                             .00461596991054300078e-5l,
                                             .00104166797027146217e-5l,
                                             .00023715009995921222e-5l,
                                             .00005439284068471390e-5l,
                                             .00001255489864097987e-5l,
                                             .291381803663201e-10l,
                                             .067949421808797e-10l,
                                             .015912343331569e-10l,
                                             .003740250585245e-10l,
                                             .000882087762421e-10l,
                                             .000208650897725e-10l,
                                             .000049488041039e-10l,
                                             .000011766394740e-10l,
                                             .2803855725e-15l,
                                             .0669506638e-15l,
                                             .0160165495e-15l,
                                             .0038382583e-15l,
                                             .0009212851e-15l,
                                             .0002214615e-15l,
                                             .0000533091e-15l,
                                             .0000128488e-15l,
                                             .31006e-20l,
                                             .07491e-20l,
                                             .01812e-20l,
                                             .00439e-20l,
                                             .00106e-20l,
                                             .00026e-20l,
                                             .00006e-20l,
                                             .00002e-20l};
  return erfinvChebyshevSeries(p, p * p / 0.32 - 1.0, xi, xiSize);
}

/**
 * @fn WLambert
 * @param x
 * @param w0
 * @param epsilon
 * @return
 */
double WLambert(double x, double w0, double epsilon)
{
  double w = w0;
  double step = 0;
  do
  {
    double ew = std::exp(w);
    double wew = w * ew;
    double numerator1 = wew - x;
    double wp1 = w + 1;
    double denominator1 = ew * wp1;
    double numerator2 = (w + 2) * numerator1;
    double denominator2 = 2 * wp1;
    step = numerator2 / denominator2;
    step = numerator1 / (denominator1 - step);
    w -= step;
  } while(std::fabs(step) > epsilon);
  return w;
}
} // namespace

int randlib::RandMath::sign(double x)
{
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

double randlib::RandMath::atan(double x)
{
  /// For small absolute values we use standard technique
  /// Otherwise we use relation
  /// atan(x) = +/-π/2 - atan(1/x)
  /// to avoid numeric problems
  if(x == 0.0)
    return 0.0;
  if(x > 1.0)
    return M_PI_2 - std::atan(1.0 / x);
  return (x < -1.0) ? -M_PI_2 - std::atan(1.0 / x) : std::atan(x);
}

double randlib::RandMath::softplus(double x)
{
  if(x < 20.0)
    return std::log1pl(std::exp(x));
  return (x < 35.0) ? x + std::exp(-x) : x;
}

double randlib::RandMath::log1mexp(double x)
{
  return (x < -M_LN2) ? std::log1pl(-std::exp(x)) : std::log(-std::expm1l(x));
}

double randlib::RandMath::logexpm1l(double x)
{
  if(x < 20.0)
    return std::log(std::expm1l(x));
  return (x < 35.0) ? x - std::exp(-x) : x;
}

double randlib::RandMath::log2mexp(double x)
{
  return std::log1pl(-std::expm1l(x));
}

double randlib::RandMath::erfinv(double p)
{
  /// Consider special cases
  if(p < 0.0)
    return -randlib::RandMath::erfinv(-p);
  if(p > 1.0)
    throw std::invalid_argument("Argument p should be in interval [-1, 1]");
  if(p == 1.0)
    return INFINITY;
  if(p == 0.0)
    return 0.0;
  if(p < 0.8)
    return erfinvAux4(p);
  /// Handle tails
  double beta = std::sqrt(-std::log1pl(-p * p));
  if(p < 0.9975)
    return erfinvAux3(beta);
  return (1.0 - p < 5e-16) ? erfinvAux1(beta) : erfinvAux2(beta);
}

double randlib::RandMath::erfcinv(double p)
{
  /// Consider special cases
  if(p > 1.0)
    return -randlib::RandMath::erfcinv(2.0 - p);
  if(p == 0.0)
    return INFINITY;
  if(p == 1.0)
    return 0.0;
  if(p > 0.2)
    return erfinvAux4(1.0 - p);
  double pSq = p * p, p2 = 2 * p;
  double beta = std::sqrt(-std::log(p2 - pSq));
  if(p > 0.0025)
    return erfinvAux3(beta);
  return (p > 5e-16) ? erfinvAux2(beta) : erfinvAux1(beta);
}

long double randlib::RandMath::logBesselI(double nu, double x)
{
  if(x < 0)
  {
    double roundNu = std::round(nu);
    bool nuIsInt = areClose(nu, roundNu);
    if(nuIsInt)
    {
      int nuInt = roundNu;
      return (nuInt % 2) ? NAN : randlib::RandMath::logBesselI(nu, -x);
    }
    return -INFINITY;
  }

  if(x == 0)
  {
    if(nu == 0)
      return 0.0;
    double roundNu = std::round(nu);
    bool nuIsInt = areClose(nu, roundNu);
    return (nu > 0 || nuIsInt) ? -INFINITY : INFINITY;
  }

  if(std::fabs(nu) == 0.5)
  {
    /// log(sinh(x)) or log(cosh(x))
    long double y = x - 0.5 * (M_LN2 + M_LNPI + std::log(x));
    y += (nu > 0) ? RandMath::softplus(-2 * x) : RandMath::log1mexp(-2 * x);
    return y;
  }

  if(nu < 0)
  {
    /// I(ν, x) = I(−ν, x) - 2 / π sin(πν) K(ν, x)
    long double besseli = std::cyl_bessel_il(-nu, x);
    long double sinPiNu = -std::sin(M_PI * nu);
    long double y = 0;
    if(sinPiNu == 0 || RandMath::areClose(nu, std::round(nu)))
      y = besseli;
    else
    {
      long double besselk = std::cyl_bessel_kl(-nu, x);
      y = besseli - M_2_PI * sinPiNu * besselk;
    }
    return (y <= 0) ? -INFINITY : std::log(y);
  }

  long double besseli = std::cyl_bessel_il(nu, x); // TODO: expand Hankel asymptotic expansions
  return std::isfinite(besseli) ? std::log(besseli) : x - 0.5 * (M_LN2 + M_LNPI + std::log(x));
}

long double randlib::RandMath::logBesselK(double nu, double x)
{
  if(nu < 0.0)
    return NAN; /// K(-ν, x) = -K(ν, x) < 0

  if(x == 0.0)
    return INFINITY;

  long double besselk = 0;
  if(nu == 0.5 || (besselk = std::cyl_bessel_kl(nu, x)) == 0)
    return 0.5 * (M_LNPI - M_LN2 - std::log(x)) - x;

  if(!std::isfinite(besselk)) // TODO: expand Hankel asymptotic expansions
    return (nu == 0) ? std::log(-std::log(x)) : std::lgammal(nu) - M_LN2 - nu * std::log(0.5 * x);

  return std::log(besselk);
}

double randlib::RandMath::W0Lambert(double x, double epsilon)
{
  double w = 0;
  if(x < -M_1_E)
    throw std::invalid_argument("Argument x should be greater than -1/e, but it's equal to " + std::to_string(x));
  if(x > 10)
  {
    double logX = std::log(x);
    double loglogX = std::log(logX);
    w = logX - loglogX;
  }
  return WLambert(x, w, epsilon);
}

double randlib::RandMath::Wm1Lambert(double x, double epsilon)
{
  double w = -2;
  if(x < -M_1_E || x > 0)
    throw std::invalid_argument("Argument x should be greater than -1/e and smaller or equal to 0, but it's equal to " + std::to_string(x));
  if(x > -0.1)
  {
    double logmX = std::log(-x);
    double logmlogmX = std::log(-logmX);
    w = logmX - logmlogmX;
  }
  return WLambert(x, w, epsilon);
}
