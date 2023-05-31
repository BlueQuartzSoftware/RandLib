#pragma once

#include "UnivariateDistribution.hpp"

namespace
{
/**
 * @fn areClose
 * @param a
 * @param b
 * @param eps
 * @return |a - b| < eps * max(a, b)
 */
template <typename RealType>
bool areClose(RealType a, RealType b, RealType eps = 1e-6)
{
  if(a == b)
    return true;
  RealType fa = std::fabs(a);
  RealType fb = std::fabs(b);
  return std::fabs(b - a) < eps * std::max(fa, fb);
}

/**
 * @fn sign
 * @param x
 * @return sign of x
 */
template <typename RealType>
int sign(RealType x)
{
  return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

template <typename RealType>
RealType adaptiveSimpsonsAux(const std::function<double(double)>& funPtr, RealType a, RealType b, RealType epsilon, RealType S, RealType fa, RealType fb, RealType fc, int bottom)
{
  RealType c = .5 * (a + b), h = (b - a) / 12.0;
  RealType d = .5 * (a + c), e = .5 * (c + b);
  RealType fd = funPtr(d), fe = funPtr(e);
  RealType Sleft = h * (fa + 4 * fd + fc);
  RealType Sright = h * (fc + 4 * fe + fb);
  RealType S2 = Sleft + Sright;
  if(bottom <= 0 || std::fabs(S2 - S) <= 15.0 * epsilon)
  {
    return S2 + (S2 - S) / 15.0;
  }
  epsilon *= .5;
  --bottom;

  return adaptiveSimpsonsAux(funPtr, a, c, epsilon, Sleft, fa, fc, fd, bottom) + adaptiveSimpsonsAux(funPtr, c, b, epsilon, Sright, fc, fb, fe, bottom);
}

/**
 * @fn integral
 * @param funPtr integrand
 * @param a lower boundary
 * @param b upper boundary
 * @param epsilon tolerance
 * @param maxRecursionDepth how deep should the algorithm go
 * @return
 */
template <typename RealType>
RealType integral(const std::function<double(double)>& funPtr, RealType a, RealType b, RealType epsilon = 1e-11, int maxRecursionDepth = 11)
{
  {
    if(a > b)
      return -integral(funPtr, b, a, epsilon, maxRecursionDepth);
    if(a == b)
      return 0.0;
    RealType c = .5 * (a + b), h = (b - a) / 6.0;
    RealType fa = funPtr(a), fb = funPtr(b), fc = funPtr(c);
    RealType S = h * (fa + 4 * fc + fb);
    return adaptiveSimpsonsAux(funPtr, a, b, epsilon, S, fa, fb, fc, maxRecursionDepth);
  }
}

/**
 * @fn findRootNewtonFirstOrder
 * Newton's root-finding procedure,
 * using first derivative
 * @param funPtr mapping x |-> (f(x), f'(x))
 * @param root starting point in input and such x that f(x) = 0 in output
 * @param funTol function tolerance
 * @param stepTol step tolerance
 * @return true if success, false otherwise
 */
template <typename RealType>
bool findRootNewtonFirstOrder(const std::function<DoublePair(RealType)>& funPtr, RealType& root, long double funTol = 1e-10, long double stepTol = 1e-6)
{
  /// Sanity check
  funTol = funTol > MIN_POSITIVE ? funTol : MIN_POSITIVE;
  stepTol = stepTol > MIN_POSITIVE ? stepTol : MIN_POSITIVE;
  static constexpr int MAX_ITER = 1e5;
  static constexpr double MAX_STEP = 10;
  int iter = 0;
  double step = stepTol + 1;
  DoublePair y = funPtr(root);
  double fun = y.first;
  double grad = y.second;
  if(std::fabs(fun) < MIN_POSITIVE)
    return true;
  do
  {
    double alpha = 1.0;
    double oldRoot = root;
    double oldFun = fun;
    step = std::min(MAX_STEP, std::max(-MAX_STEP, fun / grad));
    do
    {
      root = oldRoot - alpha * step;
      y = funPtr(root);
      fun = y.first;
      grad = y.second;
      if(std::fabs(fun) < MIN_POSITIVE)
        return true;
      alpha *= 0.5;
    } while((std::fabs(grad) <= MIN_POSITIVE || std::fabs(oldFun) < std::fabs(fun)) && alpha > 0);
    /// Check convergence criteria
    double diffX = std::fabs(root - oldRoot);
    double relDiffX = std::fabs(diffX / oldRoot);
    if(std::min(diffX, relDiffX) < stepTol)
    {
      double diffY = fun - oldFun;
      double relDiffY = std::fabs(diffY / oldFun);
      if(std::min(std::fabs(fun), relDiffY) < funTol)
        return true;
    }
  } while(++iter < MAX_ITER);
  return false;
}

/**
 * @fn findRoot
 * Brent's root-finding procedure
 * @param funPtr mapping x |-> f(x)
 * @param a lower boundary
 * @param b upper boundary
 * @param root starting point and such x that f(x) = 0
 * @param epsilon tolerance
 * @return true if success, false otherwise
 */
template <typename RealType>
bool findRootBrentFirstOrder(const std::function<double(RealType)>& funPtr, RealType a, RealType b, RealType& root, long double epsilon = 1e-8)
{
  /// Sanity check
  epsilon = epsilon > MIN_POSITIVE ? epsilon : MIN_POSITIVE;
  double fa = funPtr(a);
  if(fa == 0)
  {
    root = a;
    return true;
  }
  double fb = funPtr(b);
  if(fb == 0)
  {
    root = b;
    return true;
  }
  if(fa * fb > 0)
  {
    /// error - the root is not bracketed
    return false;
  }
  if(std::fabs(fa) < std::fabs(fb))
  {
    std::swap(a, b);
    std::swap(fa, fb);
  }
  double c = a, fc = fa;
  bool mflag = true;
  double s = b, fs = 1, d = 0;
  while(std::fabs(b - a) > epsilon)
  {
    if(!areClose<RealType>(fc, fa) && !areClose<RealType>(fb, fc))
    {
      /// inverse quadratic interpolation
      double numerator = a * fb * fc;
      double denominator = (fa - fb) * (fa - fc);
      s = numerator / denominator;
      numerator = b * fa * fc;
      denominator = (fb - fa) * (fb - fc);
      s += numerator / denominator;
      numerator = c * fa * fb;
      denominator = (fc - fa) * (fc - fb);
      s += numerator / denominator;
    }
    else
    {
      /// secant method
      s = b - fb * (b - a) / (fb - fa);
    }
    double absDiffSB2 = std::fabs(s - b);
    absDiffSB2 += absDiffSB2;
    double absDiffBC = std::fabs(b - c);
    double absDiffCD = std::fabs(c - d);
    if(s < 0.25 * (3 * a + b) || s > b || (mflag && absDiffSB2 >= absDiffBC) || (!mflag && absDiffSB2 >= absDiffCD) || (mflag && absDiffBC < epsilon) || (!mflag && absDiffCD < epsilon))
    {
      s = 0.5 * (a + b);
      mflag = true;
    }
    else
    {
      mflag = false;
    }
    fs = funPtr(s);
    if(std::fabs(fs) < epsilon)
    {
      root = s;
      return true;
    }
    d = c;
    c = b;
    fc = fb;
    if(fa * fs < 0)
    {
      b = s;
      fb = fs;
    }
    else
    {
      a = s;
      fa = fs;
    }
    if(std::fabs(fa) < std::fabs(fb))
    {
      std::swap(a, b);
      std::swap(fa, fb);
    }
  }
  root = (std::fabs(fs) < std::fabs(fb)) ? s : b;
  return true;
}

/**
 * @fn parabolicMinimum
 * @param a < b < c
 * @param fa f(a)
 * @param fb f(b)
 * @param fc f(c)
 * @return minimum of interpolated parabola
 */
template <typename RealType>
double parabolicMinimum(RealType a, RealType b, RealType c, double fa, double fb, double fc)
{
  RealType bma = b - a, cmb = c - b;
  RealType aux1 = bma * (fb - fc);
  RealType aux2 = cmb * (fb - fa);
  RealType numerator = bma * aux1 - cmb * aux2;
  RealType denominator = aux1 + aux2;
  return b - 0.5 * numerator / denominator;
}

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
} // namespace

/**
 * @brief The ContinuousDistribution class <BR>
 * Abstract class for all continuous distributions
 */
template <typename RealType, class Engine = JLKiss64RandEngine>
class RANDLIB_EXPORT ContinuousDistribution : virtual public UnivariateDistribution<RealType, Engine>
{
  static_assert(std::is_floating_point_v<RealType>, "Continuous distribution supports only floating-point types");

protected:
  ContinuousDistribution() = default;
  virtual ~ContinuousDistribution() = default;

public:
  //-------------------------------------------------------------------------------------------
  // PURE VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn logf
   * @param x
   * @return logarithm of probability density function
   */
  virtual double logf(const RealType& x) const = 0;

  //-------------------------------------------------------------------------------------------
  // VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn f
   * @param x
   * @return probability density function
   */
  virtual double f(const RealType& x) const
  {
    return std::exp(this->logf(x));
  }

  //-------------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //-------------------------------------------------------------------------------------------

  /**
   * @fn ProbabilityDensityFunction
   * fill vector y with f(x)
   * @param x
   * @param y
   */
  void ProbabilityDensityFunction(const std::vector<RealType>& x, std::vector<double>& y) const
  {
    for(size_t i = 0; i != x.size(); ++i)
      y[i] = this->f(x[i]);
  }

  /**
   * @fn LogProbabilityDensityFunction
   * fill vector y with logf(x)
   * @param x
   * @param y
   */
  void LogProbabilityDensityFunction(const std::vector<RealType>& x, std::vector<double>& y) const
  {
    for(size_t i = 0; i != x.size(); ++i)
      y[i] = this->logf(x[i]);
  }

  RealType Mode() const override
  {
    RealType guess = this->Mean(); /// good starting point
    if(!std::isfinite(guess))
      guess = this->Median(); /// this shouldn't be nan or inf
    RealType root = 0;
    ::findMin<RealType>([this](const RealType& x) { return -this->logf(x); }, guess, root);
    return root;
  }

  double Hazard(const RealType& x) const override
  {
    if(x < this->MinValue())
      return 0.0; /// 0/1
    if(x > this->MaxValue())
      return NAN; /// 0/0
    return this->f(x) / this->S(x);
  }

  double LikelihoodFunction(const std::vector<RealType>& sample) const override
  {
    return std::exp(LogLikelihoodFunction(sample));
  }

  double LogLikelihoodFunction(const std::vector<RealType>& sample) const override
  {
    long double res = 0.0;
    for(const RealType& var : sample)
      res += this->logf(var);
    return res;
  }

protected:
  //-------------------------------------------------------------------------------------------
  // NON-VIRTUAL
  //-------------------------------------------------------------------------------------------

  RealType quantileImpl(double p, RealType initValue) const override
  {
    static constexpr double SMALL_P = 1e-5;
    if(p < SMALL_P)
    {
      /// for small p we use logarithmic scale
      double logP = std::log(p);
      if(!::findRootNewtonFirstOrder<RealType>(
             [this, logP](const RealType& x) {
               double logCdf = std::log(this->F(x)), logPdf = this->logf(x);
               double first = logCdf - logP;
               double second = std::exp(logPdf - logCdf);
               return DoublePair(first, second);
             },
             initValue))
        throw std::runtime_error("Continuous distribution: failure in numeric procedure");
      return initValue;
    }

    if(this->SupportType() == FINITE_T)
    {
      if(!::findRootBrentFirstOrder<RealType>([this, p](const RealType& x) { return this->F(x) - p; }, this->MinValue(), this->MaxValue(), initValue))
        throw std::runtime_error("Continuous distribution: failure in numeric procedure");
      return initValue;
    }

    if(!::findRootNewtonFirstOrder<RealType>(
           [this, p](const RealType& x) {
             double first = this->F(x) - p;
             double second = this->f(x);
             return DoublePair(first, second);
           },
           initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  RealType quantileImpl(double p) const override
  {
    /// We use quantile from sample as an initial guess
    static constexpr int SAMPLE_SIZE = 128;
    static std::vector<RealType> sample(SAMPLE_SIZE);
    this->Sample(sample);
    int index = p * SAMPLE_SIZE;
    if(index == 0.0)
      return this->quantileImpl(p, *std::min_element(sample.begin(), sample.end()));
    std::nth_element(sample.begin(), sample.begin() + index, sample.end());
    return this->quantileImpl(p, sample[index]);
  }

  RealType quantileImpl1m(double p, RealType initValue) const override
  {
    static constexpr double SMALL_P = 1e-5;
    if(p < SMALL_P)
    {
      /// for small p we use logarithmic scale
      double logP = std::log(p);
      if(!::findRootNewtonFirstOrder<RealType>(
             [this, logP](const RealType& x) {
               double logCcdf = std::log(this->S(x)), logPdf = this->logf(x);
               double first = logP - logCcdf;
               double second = std::exp(logPdf - logCcdf);
               return DoublePair(first, second);
             },
             initValue))
        throw std::runtime_error("Continuous distribution: failure in numeric procedure");
      return initValue;
    }

    if(this->SupportType() == FINITE_T)
    {
      if(!::findRootBrentFirstOrder<RealType>([this, p](const RealType& x) { return this->S(x) - p; }, this->MinValue(), this->MaxValue(), initValue))
        throw std::runtime_error("Continuous distribution: failure in numeric procedure");
      return initValue;
    }

    if(!::findRootNewtonFirstOrder<RealType>(
           [this, p](const RealType& x) {
             double first = p - this->S(x);
             double second = this->f(x);
             return DoublePair(first, second);
           },
           initValue))
      throw std::runtime_error("Continuous distribution: failure in numeric procedure");
    return initValue;
  }

  RealType quantileImpl1m(double p) const override
  {
    /// We use quantile from sample as an initial guess
    static constexpr int SAMPLE_SIZE = 128;
    static std::vector<RealType> sample(SAMPLE_SIZE);
    this->Sample(sample);
    int index = p * SAMPLE_SIZE;
    if(index == 0.0)
      return this->quantileImpl1m(p, *std::max_element(sample.begin(), sample.end()));
    std::nth_element(sample.begin(), sample.begin() + index, sample.end(), std::greater<>());
    return this->quantileImpl1m(p, sample[index]);
  }

  long double ExpectedValue(const std::function<double(RealType)>& funPtr, RealType minPoint, RealType maxPoint) const override
  {
    /// attempt to calculate expected value by numerical method
    /// use for distributions w/o explicit formula
    RealType lowerBoundary = minPoint, upperBoundary = maxPoint;
    if(this->isRightBounded())
      lowerBoundary = std::max(minPoint, lowerBoundary);
    if(this->isLeftBounded())
      upperBoundary = std::min(maxPoint, upperBoundary);

    if(lowerBoundary >= upperBoundary)
      return 0.0;

    bool isLeftBoundFinite = std::isfinite(lowerBoundary), isRightBoundFinite = std::isfinite(upperBoundary);

    /// Integrate on finite interval [a, b]
    if(isLeftBoundFinite && isRightBoundFinite)
    {
      return ::integral(
          [this, funPtr](double x) {
            double y = funPtr(x);
            return (y == 0.0) ? 0.0 : y * f(x);
          },
          lowerBoundary, upperBoundary);
    }

    /// Integrate on semifinite interval [a, inf)
    if(isLeftBoundFinite)
    {
      return ::integral(
          [this, funPtr, lowerBoundary](double x) {
            if(x >= 1.0)
              return 0.0;
            double denom = 1.0 - x;
            double t = lowerBoundary + x / denom;
            double y = funPtr(t);
            if(y == 0.0)
              return 0.0;
            y *= this->f(t);
            denom *= denom;
            return y / denom;
          },
          0.0, 1.0);
    }

    /// Integrate on semifinite intervale (-inf, b]
    if(isRightBoundFinite)
    {
      return ::integral(
          [this, funPtr, upperBoundary](double x) {
            if(x <= 0.0)
              return 0.0;
            double t = upperBoundary - (1.0 - x) / x;
            double y = funPtr(t);
            if(y == 0.0)
              return 0.0;
            y *= this->f(t);
            return y / (x * x);
          },
          0.0, 1.0);
    }

    /// Infinite case
    return ::integral(
        [this, funPtr](double x) {
          if(std::fabs(x) >= 1.0)
            return 0.0;
          double x2 = x * x;
          double denom = 1.0 - x2;
          double t = x / denom;
          double y = funPtr(t);
          if(y == 0.0)
            return 0.0;
          y *= this->f(t);
          denom *= denom;
          return y * (1.0 + x2) / denom;
        },
        -1.0, 1.0);
  }
};

/**
 * @brief The UniformRand class <BR>
 * Uniform continuous distribution
 *
 * f(x | a, b) = 1 / (b - a) for a < x < b
 *
 * Notation: X ~ U(a, b)
 *
 * Related distributions: <BR>
 * X ~ B(1, 1, a, b) <BR>
 * (X - a) / (b - a) ~ IH(1)
 */
template <typename RealType = double, class Engine = JLKiss64RandEngine>
class RANDLIB_EXPORT UniformRand : public ContinuousDistribution<RealType, Engine>
{
  static_assert(std::is_floating_point_v<RealType>, "Continuous distribution supports only floating-point types");

  RealType a = 0;      ///< min bound
  RealType b = 1;      ///< max bound
  RealType bma = 1;    ///< b-a
  RealType bmaInv = 1; ///< 1/(b-a)
  RealType logbma = 0; ///< log(b-a)

  /**
   * @fn SetSupport
   * @param minValue a
   * @param maxValue b
   */
  void SetSupport(RealType minValue, RealType maxValue)
  {
    if(minValue >= maxValue)
      throw std::invalid_argument("Beta distribution: minimum value should be "
                                  "smaller than maximum value");

    a = minValue;
    b = maxValue;
    bma = b - a;
    bmaInv = 1.0 / bma;
    logbma = std::log(bma);
  }

public:
  UniformRand(RealType minValue = 0, RealType maxValue = 1)
  : UnivariateDistribution<RealType, Engine>()
  {
    SetSupport(minValue, maxValue);
  }

  std::string Name() const override
  {
    return "Uniform(" + this->toStringWithPrecision(MinValue()) + ", " + this->toStringWithPrecision(MaxValue()) + ")";
  }

  SUPPORT_TYPE SupportType() const override
  {
    return FINITE_T;
  }
  RealType MinValue() const override
  {
    return this->a;
  }
  RealType MaxValue() const override
  {
    return this->b;
  }

  double f(const RealType& x) const override
  {
    return (x < this->a || x > this->b) ? 0.0 : this->bmaInv;
  }

  double logf(const RealType& x) const override
  {
    return (x < this->a || x > this->b) ? -INFINITY : -this->logbma;
  }

  double F(const RealType& x) const override
  {
    if(x < this->a)
      return 0.0;
    return (x > this->b) ? 1.0 : this->bmaInv * (x - this->a);
  }

  double S(const RealType& x) const override
  {
    if(x < this->a)
      return 1.0;
    return (x > this->b) ? 0.0 : this->bmaInv * (this->b - x);
  }

  RealType Variate() const override
  {
    return this->a + StandardVariate(this->localRandGenerator) * this->bma;
  }

  /**
   * @fn StandardVariate
   * @param randGenerator
   * @return a random number on interval (0,1) if no preprocessors are specified
   */
  static RealType StandardVariate(BasicRandGenerator<Engine>& randGenerator = ProbabilityDistribution<RealType, Engine>::staticRandGenerator)
  {
#ifdef RANDLIB_UNIDBL
    /// generates this->a random number on [0,1) with 53-bit resolution, using 2 32-bit integer variate
    double x;
    unsigned int a, b;
    this->a = randGenerator.Variate() >> 6; /// Upper 26 bits
    b = randGenerator.Variate() >> 5;       /// Upper 27 bits
    x = (this->a * 134217728.0 + this->b) / 9007199254740992.0;
    return x;
#elif defined(RANDLIB_JLKISS64)
    /// generates this->a random number on [0,1) with 53-bit resolution, using 64-bit integer variate
    double x;
    unsigned long long this->a = randGenerator.Variate();
    this->a = (this->a >> 12) | 0x3FF0000000000000ULL; /// Take upper 52 bit
    *(reinterpret_cast<unsigned long long*>(&x)) = a;  /// Make this->a double from bits
    return x - 1.0;
#else
    RealType x = randGenerator.Variate();
    x += 0.5;
    x /= 4294967296.0;
    return x;
#endif
  }

  /**
   * @fn StandardVariateClosed
   * @param randGenerator
   * @return a random number on interval [0,1]
   */
  static RealType StandardVariateClosed(BasicRandGenerator<Engine>& randGenerator = ProbabilityDistribution<RealType, Engine>::staticRandGenerator)
  {
    RealType x = randGenerator.Variate();
    return x / 4294967295.0;
  }

  /**
   * @fn StandardVariateHalfClosed
   * @param randGenerator
   * @return a random number on interval [0,1)
   */
  static RealType StandardVariateHalfClosed(BasicRandGenerator<Engine>& randGenerator = ProbabilityDistribution<RealType, Engine>::staticRandGenerator)
  {
    RealType x = randGenerator.Variate();
    return x / 4294967296.0;
  }

  void Sample(std::vector<RealType>& outputData) const override
  {
    for(RealType& var : outputData)
      var = this->Variate();
  }

  long double Mean() const override
  {
    return 0.5 * (this->b + this->a);
  }

  long double Variance() const override
  {
    return this->bma * this->bma / 12;
  }

  RealType Median() const override
  {
    return 0.5 * (this->b + this->a);
  }

  RealType Mode() const override
  {
    /// this can be any value in [a, b]
    return 0.5 * (this->b + this->a);
  }

  long double Skewness() const override
  {
    return 0.0;
  }

  long double ExcessKurtosis() const override
  {
    return -1.2;
  }

  long double Entropy() const
  {
    return (this->b == this->a) ? -INFINITY : std::log(this->bma);
  }

  double LikelihoodFunction(const std::vector<RealType>& sample) const override
  {
    bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(this->a, sample) && this->allElementsAreNotGreaterThan(this->b, sample);
    return sampleIsInsideInterval ? std::pow(this->bma, -sample.size()) : 0.0;
  }

  double LogLikelihoodFunction(const std::vector<RealType>& sample) const override
  {
    bool sampleIsInsideInterval = this->allElementsAreNotSmallerThan(this->a, sample) && this->allElementsAreNotGreaterThan(this->b, sample);
    int sample_size = sample.size();
    return sampleIsInsideInterval ? -sample_size * this->logbma : -INFINITY;
  }

  /**
   * @fn FitMinimum
   * fit minimum with maximum-likelihood estimator if unbiased == false,
   * fit minimum using UMVU estimator otherwise
   * @param sample
   */
  void FitMinimum(const std::vector<RealType>& sample, bool unbiased = false)
  {
    if(!this->allElementsAreNotGreaterThan(this->b, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->UPPER_LIMIT_VIOLATION + this->toStringWithPrecision(this->b)));
    RealType minVar = *std::min_element(sample.begin(), sample.end());

    if(unbiased == true)
    {
      int n = sample.size();
      /// E[min] = b - n / (n + 1) * (this->b - this->a)
      RealType minVarAdj = (minVar * (n + 1) - this->b) / n;
      if(!this->allElementsAreNotSmallerThan(minVarAdj, sample))
        throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_LARGE_A + this->toStringWithPrecision(minVarAdj)));
      SetSupport(minVarAdj, this->b);
    }
    else
    {
      SetSupport(minVar, this->b);
    }
  }

  /**
   * @fn FitMaximum
   * fit maximum with maximum-likelihood estimator if unbiased == false,
   * fit maximum using UMVU estimator otherwise
   * @param sample
   */
  void FitMaximum(const std::vector<RealType>& sample, bool unbiased = false)
  {
    if(!this->allElementsAreNotSmallerThan(this->a, sample))
      throw std::invalid_argument(this->fitErrorDescription(this->WRONG_SAMPLE, this->LOWER_LIMIT_VIOLATION + this->toStringWithPrecision(this->a)));
    RealType maxVar = *std::max_element(sample.begin(), sample.end());

    if(unbiased == true)
    {
      int n = sample.size();
      /// E[max] = (this->b - this->a) * n / (n + 1) + a
      RealType maxVarAdj = (maxVar * (n + 1) - this->a) / n;
      if(!this->allElementsAreNotGreaterThan(maxVarAdj, sample))
        throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_SMALL_B + this->toStringWithPrecision(maxVarAdj)));
      SetSupport(this->a, maxVarAdj);
    }
    else
    {
      SetSupport(this->a, maxVar);
    }
  }

  /**
   * @fn Fit
   * fit support with maximum-likelihood estimator if unbiased == false,
   * fit support using UMVU estimator otherwise
   * @param sample
   */
  void Fit(const std::vector<RealType>& sample, bool unbiased = false)
  {
    double minVar = *std::min_element(sample.begin(), sample.end());
    double maxVar = *std::max_element(sample.begin(), sample.end());
    if(unbiased == true)
    {
      int n = sample.size();
      /// E[min] = b - n / (n + 1) * (this->b - this->a)
      RealType minVarAdj = (minVar * n - maxVar) / (n - 1);
      /// E[max] = (this->b - this->a) * n / (n + 1) + a
      RealType maxVarAdj = (maxVar * n - minVar) / (n - 1);
      if(!this->allElementsAreNotSmallerThan(minVarAdj, sample))
        throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_LARGE_A + this->toStringWithPrecision(minVarAdj)));
      if(!this->allElementsAreNotGreaterThan(maxVarAdj, sample))
        throw std::runtime_error(this->fitErrorDescription(this->WRONG_RETURN, TOO_SMALL_B + this->toStringWithPrecision(maxVarAdj)));
      SetSupport(minVarAdj, maxVarAdj);
    }
    else
    {
      SetSupport(minVar, maxVar);
    }
  }

private:
  RealType quantileImpl(double p) const override
  {
    return this->a + this->bma * p;
  }

  RealType quantileImpl1m(double p) const override
  {
    return this->b - this->bma * p;
  }

  std::complex<double> CFImpl(double t) const override
  {
    double cosX = std::cos(t * this->b), sinX = std::sin(t * this->b);
    double cosY = std::cos(t * this->a), sinY = std::sin(t * this->a);
    std::complex<double> numerator(cosX - cosY, sinX - sinY);
    std::complex<double> denominator(0, t * this->bma);
    return numerator / denominator;
  }

  static constexpr char TOO_LARGE_A[] = "Minimum element of the sample is smaller than lower boundary returned by method: ";
  static constexpr char TOO_SMALL_B[] = "Maximum element of the sample is greater than upper boundary returned by method: ";
};
