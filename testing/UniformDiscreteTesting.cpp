#include "RandLib/distributions/BasicRandGenerator.hpp"
#include "RandLib/distributions/UniformDiscreteRand.hpp"

using namespace RandLib;

namespace
{
template <typename IntType>
bool TestDiscreteEngines(IntType min, IntType max)
{
  unsigned long seed = 5489;
  {
    UniformDiscreteRand<IntType, JLKiss64RandEngine> uDist = UniformDiscreteRand<IntType, JLKiss64RandEngine>(min, max);
    uDist.Reseed(seed);
    std::vector<IntType> data(1000);
    uDist.Sample(data);

    for(const auto num : data)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    UniformDiscreteRand<IntType, JLKiss64RandEngine> uDist1 = UniformDiscreteRand<IntType, JLKiss64RandEngine>(min, max);
    uDist1.Reseed(seed);
    std::vector<IntType> data1(1000);
    uDist1.Sample(data1);

    for(const auto num : data1)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    for(size_t i = 0; i < data.size(); i++)
    {
      if(data[i] != data1[i])
      {
        return false;
      }
    }
  }

  {
    UniformDiscreteRand<IntType, JKissRandEngine> uDist = UniformDiscreteRand<IntType, JKissRandEngine>(min, max);
    uDist.Reseed(seed);
    std::vector<IntType> data(1000);
    uDist.Sample(data);

    for(const auto num : data)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    UniformDiscreteRand<IntType, JKissRandEngine> uDist1 = UniformDiscreteRand<IntType, JKissRandEngine>(min, max);
    uDist1.Reseed(seed);
    std::vector<IntType> data1(1000);
    uDist1.Sample(data1);

    for(const auto num : data1)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    for(size_t i = 0; i < data.size(); i++)
    {
      if(data[i] != data1[i])
      {
        return false;
      }
    }
  }

  {
    UniformDiscreteRand<IntType, PCGRandEngine> uDist = UniformDiscreteRand<IntType, PCGRandEngine>(min, max);
    uDist.Reseed(seed);
    std::vector<IntType> data(1000);
    uDist.Sample(data);

    for(const auto num : data)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    UniformDiscreteRand<IntType, PCGRandEngine> uDist1 = UniformDiscreteRand<IntType, PCGRandEngine>(min, max);
    uDist1.Reseed(seed);
    std::vector<IntType> data1(1000);
    uDist1.Sample(data1);

    for(const auto num : data1)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    for(size_t i = 0; i < data.size(); i++)
    {
      if(data[i] != data1[i])
      {
        return false;
      }
    }
  }

  return true;
}
} // namespace

int main()
{
  if(!TestDiscreteEngines<short>(-76, 89))
  {
    return 1;
  }

  if(!TestDiscreteEngines<int>(-764, 8639))
  {
    return 1;
  }

  if(!TestDiscreteEngines<long>(-7645, 86398))
  {
    return 1;
  }

  return 0;
}
