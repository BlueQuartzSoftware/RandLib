#include "MarcumMath.h"

using namespace randlib;

namespace
{
    /**
 * @fn MarcumPSeries
 * @param mu
 * @param x
 * @param y
 * @param logX log(x)
 * @param logY log(y)
 * @return series expansion for Marcum-P function
 */
    double MarcumPSeries(double mu, double x, double y, double logX, double logY)
    {
        /// ~log(2πε) for ε = 1e-16
        static constexpr double ln2piEps = -35.0;
        double lgammamu = std::lgammal(mu);
        double C = lgammamu - ln2piEps + mu;

        /// solving equation f(n) = 0
        /// to find first negleted term
        double root = std::max(0.5 * (mu * mu + 4 * x * y - mu), 1.0);
        double logXY = logX + logY;
        if(!RandMath::findRootNewtonSecondOrder<double>(
                [C, mu, logXY](double n) {
                    double npmu = n + mu;
                    double logn = std::log(n), lognpmu = std::log(npmu);
                    double first = logn - 2 - logXY;
                    first *= n;
                    first += npmu * lognpmu;
                    first -= C;
                    double second = logn + lognpmu - logXY;
                    double third = 1.0 / n + 1.0 / npmu;
                    return DoubleTriplet(first, second, third);
                },
                root))
            /// unexpected return
            throw std::runtime_error("Marcum P function: failure in numerical procedure");

        /// series expansion
        double sum = 0.0;
        /// sanity check
        int n0 = std::max(std::ceil(root), 5.0);
        double mpn0 = mu + n0;
        double P = randlib::RandMath::pgamma(mpn0, y, logY);
        double diffP = (mpn0 - 1) * logY - y - std::lgammal(mpn0);
        diffP = std::exp(diffP);
        for(int n = n0; n > 0; --n)
        {
            double term = n * logX - x - randlib::RandMath::lfact(n);
            double mupnm1 = mu + n - 1;
            term = std::exp(term) * P;
            sum += term;
            if(n % 5 == 0)
            {
                /// every 5 iterations we recalculate P and diffP
                /// in order to achieve enough accuracy
                P = randlib::RandMath::pgamma(mupnm1, y, logY);
                diffP = (mupnm1 - 1) * logY - y - std::lgammal(mupnm1);
                diffP = std::exp(diffP);
            }
            else
            {
                /// otherwise we use recurrent relations
                P += diffP;
                diffP *= mupnm1 / y;
            }
        }
        /// add the last 0-term
        double lastTerm = randlib::RandMath::lpgamma(mu, y, logY) - x;
        lastTerm = std::exp(lastTerm);
        sum += lastTerm;
        return sum;
    }

/**
 * @fn MarcumPAsymptoticForLargeXY
 * @param mu
 * @param x
 * @param y
 * @param sqrtX √x
 * @param sqrtY √y
 * @return asymptotic expansion for Marcum-P function for large x*y
 */
    double MarcumPAsymptoticForLargeXY(double mu, double x, double y, double sqrtX, double sqrtY)
    {
        double xi = 2 * sqrtX * sqrtY;
        double sigma = (y + x) / xi - 1;
        double sigmaXi = sigma * xi;
        double rho = sqrtY / sqrtX;
        double aux = std::erfc(sqrtX - sqrtY);
        double Phi = (sigma == 0) ? 0.0 : randlib::RandMath::sign(x - y) * std::sqrt(M_PI / sigma) * aux;
        double Psi0 = aux / std::sqrt(rho);
        double logXi = M_LN2 + 0.5 * std::log(x * y);
        /// sanity check
        int n0 = std::max(std::ceil(sigmaXi), 7.0);
        double sum = 0.0;
        double A1 = 1.0, A2 = 1.0;
        for(int n = 1; n <= n0; ++n)
        {
            /// change φ
            double nmHalf = n - 0.5;
            Phi *= -sigma;
            double temp = -sigmaXi - nmHalf * logXi;
            Phi += std::exp(temp);
            Phi /= nmHalf;
            /// calculate A(μ-1) and A(μ)
            double coef1 = (nmHalf - mu) * (nmHalf + mu);
            double coef2 = (nmHalf - mu + 1) * (nmHalf + mu - 1);
            double denom = -2 * n;
            A1 *= coef1 / denom;
            A2 *= coef2 / denom;
            /// compute term ψ and add it to the sum
            double Psi = Phi * (A2 - A1 / rho);
            sum += (n & 1) ? Psi : -Psi;
        }
        sum /= M_SQRT2PI;
        sum += Psi0;
        sum *= 0.5 * std::pow(rho, mu);
        return sum;
    }

/**
 * @fn MarcumPForMuSmallerThanOne
 * @param mu
 * @param x
 * @param y
 * @return
 */
    double MarcumPForMuSmallerThanOne(double mu, double x, double y, double logX, double logY)
    {
        // TODO: check Krishnamoorthy paper for alternative representation

        /// in this case we use numerical integration,
        /// however we have singularity point at 0,
        /// so we get rid of it by subtracting the function
        /// which has the same behaviour at this point
        double aux = x + mu * M_LN2 + std::lgammal(mu);
        double log2x = M_LN2 + logX;
        double I = (M_LN2 + logY) * mu - aux;
        I = std::exp(I) / mu;

        double mum1 = mu - 1.0;
        I += RandMath::integral(
                [x, log2x, mum1, aux](double t) {
                    if(t <= 0)
                        return 0.0;
                    /// Calculate log of leveling factor
                    double log2T = std::log(2 * t);
                    double exponent = mum1 * log2T;
                    exponent -= aux;

                    /// Calculate log(2f(t))
                    double logBessel = RandMath::logBesselI(mum1, 2 * std::sqrt(x * t));
                    double z = mum1 * (log2T - log2x);
                    double log2F = 0.5 * z - t - x + logBessel;

                    /// Return difference f(t) - factor
                    return std::exp(log2F) - 2 * std::exp(exponent);
                },
                0, y);
        return I;
    }

    double MarcumQIntegrand(double theta, double xi, double sqrt1pXiSq, double mu, double y)
    {
        double sinTheta = std::sin(theta);
        double theta_sinTheta = theta / sinTheta;
        double rho = std::hypot(theta_sinTheta, xi);
        double theta_sinThetapRho = theta_sinTheta + rho;
        double r = 0.5 * theta_sinThetapRho / y;
        double cosTheta = std::cos(theta);
        double psi = cosTheta * rho - sqrt1pXiSq;
        double frac = theta_sinThetapRho / (1.0 + sqrt1pXiSq);
        psi -= std::log(frac);
        double numerator = (sinTheta - theta * cosTheta) / (sinTheta * rho);
        numerator += cosTheta - r;
        numerator *= r;
        double denominator = r * r - 2.0 * r * cosTheta + 1.0;
        double f = numerator / denominator;
        return std::exp(mu * psi) * f;
    }

    double MarcumQIntergralRepresentation(double mu, double x, double y, double sqrtX, double sqrtY)
    {
        double xi = 2.0 * sqrtX * sqrtY / mu;
        double sqrt1pXiSq = std::sqrt(1.0 + xi * xi);
        double yPrime = y / mu;
        double s0 = 0.5 * (1.0 + sqrt1pXiSq) / yPrime;
        double phi = x / s0 + y * s0 - std::log(s0) * mu;
        std::function<double(double)> integrandPtr = std::bind(&MarcumQIntegrand, std::placeholders::_1, xi, sqrt1pXiSq, mu, yPrime);
        double integral = RandMath::integral(integrandPtr, -M_PI, M_PI);
        return 0.5 * std::exp(-x - y + phi) / M_PI * integral;
    }
}

double randlib::RandMath::MarcumP(double mu, double x, double y, double sqrtX, double sqrtY, double logX, double logY)
{
    /* 1 - ok
     * 2 - ok
     * 3 - no
     * 4 - no
     * 5 - not yet
     * */
    if(x < 0.0 || y <= 0.0)
        return 0.0;

    if(x < 30)
        return MarcumPSeries(mu, x, y, logX, logY);

    double xi = 2 * sqrtX * sqrtY;
    if(xi > 30 && mu * mu < 2 * xi)
        return MarcumPAsymptoticForLargeXY(mu, x, y, sqrtX, sqrtY);

    double temp = std::sqrt(4 * x + 2 * mu);
    double f1 = x + mu - temp, f2 = x + mu + temp;
    if(y > f1 && y < f2)
        // IF mu > 135
        return 1.0 - MarcumQIntergralRepresentation(mu, x, y, sqrtX, sqrtY);

    // TODO: implement the rest techniques

    double mum1 = mu - 1;
    return RandMath::integral(
            [mum1, logX, x](double t) {
                if(t < 0.0)
                    return 0.0;
                if(t == 0.0)
                    return (mum1 == 0) ? 0.5 * std::exp(-x) : 0.0;
                double logBesseli = RandMath::logBesselI(mum1, 2 * std::sqrt(x * t));
                double z = 0.5 * mum1 * (std::log(t) - logX);
                double h = t + x;
                return std::exp(logBesseli + z - h);
            },
            0, y);
}

double randlib::RandMath::MarcumP(double mu, double x, double y)
{
    double sqrtX = std::sqrt(x), sqrtY = std::sqrt(y);
    double logX = std::log(x), logY = std::log(y);
    return randlib::RandMath::MarcumP(mu, x, y, sqrtX, sqrtY, logX, logY);
}

double randlib::RandMath::MarcumQ(double mu, double x, double y, double sqrtX, double sqrtY, double logX, double logY)
{
    // TODO: implement and use, when mu + x > y
    return 1.0 - randlib::RandMath::MarcumP(mu, x, y, sqrtX, sqrtY, logX, logY);
}

double randlib::RandMath::MarcumQ(double mu, double x, double y)
{
    double sqrtX = std::sqrt(x), sqrtY = std::sqrt(y);
    double logX = std::log(x), logY = std::log(y);
    return randlib::RandMath::MarcumQ(mu, x, y, sqrtX, sqrtY, logX, logY);
}