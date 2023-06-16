#include "RandLib/distributions/BasicRandGenerator.hpp"
#include "RandLib/distributions/UniformRand.hpp"

using namespace RandLib;

namespace
{
template <typename RealType>
bool TestContinuousEngines(RealType min, RealType max)
{
  unsigned long seed = 5489;
  {
    UniformRand<RealType, JKissRandEngine> uDist = UniformRand<RealType, JKissRandEngine>(min, max);
    uDist.Reseed(seed);
    std::vector<RealType> data(1000);
    uDist.Sample(data);

    for(const auto num : data)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    UniformRand<RealType, JKissRandEngine> uDist1 = UniformRand<RealType, JKissRandEngine>(min, max);
    uDist1.Reseed(seed);
    std::vector<RealType> data1(1000);
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
    UniformRand<RealType, PCGRandEngine> uDist = UniformRand<RealType, PCGRandEngine>(min, max);
    uDist.Reseed(seed);
    std::vector<RealType> data(1000);
    uDist.Sample(data);

    for(const auto num : data)
    {
      if(num < min || num > max)
      {
        return false;
      }
    }

    UniformRand<RealType, PCGRandEngine> uDist1 = UniformRand<RealType, PCGRandEngine>(min, max);
    uDist1.Reseed(seed);
    std::vector<RealType> data1(1000);
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
  if(!TestContinuousEngines<float>(-76.056, 89.3456))
  {
    return 1;
  }

  if(!TestContinuousEngines<double>(-764.056265234, 8639.345796))
  {
    return 1;
  }

  return 0;
}
