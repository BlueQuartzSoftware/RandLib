#pragma once

#include "RandLib_global.h"

#include <cstddef>
#include <ctime>
#include <thread>
#include <type_traits>

/**
 * @brief The RandEngine class
 */
class RandEngine
{
protected:
  /**
   * @fn mix
   * Robert Jenkins' 96 bit mix function
   * @param a
   * @param b
   * @param c
   * @return mixing part of the hash function
   */
  static unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
  {
    a = a - b;
    a = a - c;
    a = a ^ (c >> 13);
    b = b - c;
    b = b - a;
    b = b ^ (a << 8);
    c = c - a;
    c = c - b;
    c = c ^ (b >> 13);
    a = a - b;
    a = a - c;
    a = a ^ (c >> 12);
    b = b - c;
    b = b - a;
    b = b ^ (a << 16);
    c = c - a;
    c = c - b;
    c = c ^ (b >> 5);
    a = a - b;
    a = a - c;
    a = a ^ (c >> 3);
    b = b - c;
    b = b - a;
    b = b ^ (a << 10);
    c = c - a;
    c = c - b;
    c = c ^ (b >> 15);
    return c;
  }

  /**
   * @fn getRandomSeed
   * @return seed as a mix of time and thread id
   */
  static unsigned long getRandomSeed()
  {
    static unsigned long dummy = 123456789;
    return mix(std::time(nullptr), std::hash<std::thread::id>()(std::this_thread::get_id()), ++dummy);
  }

public:
  RandEngine() = default;

  virtual ~RandEngine() = default;

  virtual unsigned long long MinValue() const = 0;

  virtual unsigned long long MaxValue() const = 0;

  virtual void Reseed(unsigned long seed) = 0;

  virtual unsigned long long Next() = 0;
};

/**
 * @brief The JKissRandEngine class
 */
class JKissRandEngine : public RandEngine
{
  unsigned int X{};
  unsigned int C{};
  unsigned int Y{};
  unsigned int Z{};

public:
  JKissRandEngine()
  {
    this->Reseed(getRandomSeed());
  }

  unsigned long long MinValue() const override
  {
    return 0;
  }

  unsigned long long MaxValue() const override
  {
    return 4294967295UL;
  }

  void Reseed(unsigned long seed) override
  {
    X = 123456789 ^ seed;
    C = 6543217;
    Y = 987654321;
    Z = 43219876;
  }

  unsigned long long Next() override
  {
    unsigned long long t = 698769069ULL * Z + C;

    X *= 69069;
    X += 12345;

    Y ^= Y << 13;
    Y ^= Y >> 17;
    Y ^= Y << 5;

    C = t >> 32;
    Z = t;

    return X + Y + Z;
  }
};

/**
 * @brief The JLKiss64RandEngine class
 */
class JLKiss64RandEngine : public RandEngine
{
  unsigned long long X{};
  unsigned long long Y{};
  unsigned int Z1{};
  unsigned int Z2{};
  unsigned int C1{};
  unsigned int C2{};

public:
  JLKiss64RandEngine()
  {
    this->Reseed(getRandomSeed());
  }

  unsigned long long MinValue() const override
  {
    return 0;
  }

  unsigned long long MaxValue() const override
  {
    return 18446744073709551615ULL;
  }

  void Reseed(unsigned long seed) override
  {
    X = 123456789123ULL ^ seed;
    Y = 987654321987ULL;
    Z1 = 43219876;
    Z2 = 6543217;
    C1 = 21987643;
    C2 = 1732654;
  }

  unsigned long long Next() override
  {
    X = 1490024343005336237ULL * X + 123456789;
    Y ^= Y << 21;
    Y ^= Y >> 17;
    Y ^= Y << 30;

    unsigned long long t = 4294584393ULL * Z1 + C1;
    C1 = t >> 32;
    Z1 = t;
    t = 4246477509ULL * Z2 + C2;
    C2 = t >> 32;
    Z2 = t;
    return X + Y + Z1 + (static_cast<unsigned long long>(Z2) << 32);
  }
};

/**
 * @brief The PCGRandEngine class
 * Random number generator, taken from http://www.pcg-random.org/
 */
class PCGRandEngine : public RandEngine
{
  unsigned long long state{};
  unsigned long long inc{};

public:
  PCGRandEngine()
  {
    this->Reseed(getRandomSeed());
  }

  unsigned long long MinValue() const override
  {
    return 0;
  }

  unsigned long long MaxValue() const override
  {
    return 4294967295UL;
  }

  void Reseed(unsigned long seed) override
  {
    state = seed;
    inc = seed;
  }

  unsigned long long Next() override
  {
    unsigned long long oldstate = state;
    state = oldstate * 6364136223846793005ULL + (inc | 1);
    unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    unsigned int rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }
};

/**
 * @brief The BasicRandGenerator class
 * Class for generators of random number, evenly spreaded from 0 to some integer value
 */
template <class Engine = JLKiss64RandEngine>
class BasicRandGenerator
{
  static_assert(std::is_base_of<RandEngine, Engine>::value, "Engine must be a descendant of RandEngine");

  Engine engine{};

  /**
   * @fn getDecimals
   * @param value
   * @return decimals of given value
   */
  static size_t getDecimals(unsigned long long value)
  {
    size_t num = 0;
    unsigned long long maxRand = value;
    while(maxRand != 0)
    {
      ++num;
      maxRand >>= 1;
    }
    return num;
  }

public:
  BasicRandGenerator(){};

  ~BasicRandGenerator(){};

  unsigned long long Variate()
  {
    return engine.Next();
  }

  size_t maxDecimals()
  {
    return getDecimals(engine.MaxValue());
  }

  unsigned long long MaxValue()
  {
    return engine.MaxValue();
  }

  void Reseed(unsigned long seed)
  {
    engine.Reseed(seed);
  }
};
