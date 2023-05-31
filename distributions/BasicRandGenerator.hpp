#pragma once

#include "RandLib_global.h"
#include <cstddef>
#include <type_traits>

/**
 * @brief The RandEngine class
 */
class RANDLIB_EXPORT RandEngine
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
  static unsigned long mix(unsigned long a, unsigned long b, unsigned long c);
  /**
   * @fn getRandomSeed
   * @return seed as a mix of time and thread id
   */
  static unsigned long getRandomSeed();

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
class RANDLIB_EXPORT JKissRandEngine : public RandEngine
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
  void Reseed(unsigned long seed) override;
  unsigned long long Next() override;
};

/**
 * @brief The JLKiss64RandEngine class
 */
class RANDLIB_EXPORT JLKiss64RandEngine : public RandEngine
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
  void Reseed(unsigned long seed) override;
  unsigned long long Next() override;
};

/**
 * @brief The PCGRandEngine class
 * Random number generator, taken from http://www.pcg-random.org/
 */
class RANDLIB_EXPORT PCGRandEngine : public RandEngine
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
  void Reseed(unsigned long seed) override;
  unsigned long long Next() override;
};

/**
 * @brief The BasicRandGenerator class
 * Class for generators of random number, evenly spreaded from 0 to some integer value
 */
template <class Engine = JLKiss64RandEngine>
class  BasicRandGenerator
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
  BasicRandGenerator() {};
  ~BasicRandGenerator() {};

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
