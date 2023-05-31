#include "distributions/UniformRand.hpp"
#include "distributions/UniformDiscreteRand.hpp"
#include "distributions/BasicRandGenerator.hpp"

namespace
{
    template<typename RealType>
    bool TestContinuousEngines(RealType min, RealType max)
    {
        unsigned long seed = 5489;
        {
            UniformRand<RealType, JLKiss64RandEngine> uDist = UniformRand<RealType, JLKiss64RandEngine>(min, max);
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

            UniformRand<RealType, JLKiss64RandEngine> uDist1 = UniformRand<RealType, JLKiss64RandEngine>(min, max);
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

    template<typename IntType>
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
}

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