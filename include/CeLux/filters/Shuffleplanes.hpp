#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Shuffleplanes : public FilterBase {
public:
    /**
     * Shuffle video planes.
     */
    /**
     * Index of the input plane to be used as the first output plane 
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap0(int value);
    int getMap0() const;

    /**
     * Index of the input plane to be used as the second output plane 
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMap1(int value);
    int getMap1() const;

    /**
     * Index of the input plane to be used as the third output plane 
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setMap2(int value);
    int getMap2() const;

    /**
     * Index of the input plane to be used as the fourth output plane 
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setMap3(int value);
    int getMap3() const;

    Shuffleplanes(int map0 = 0, int map1 = 1, int map2 = 2, int map3 = 3);
    virtual ~Shuffleplanes();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int map0_;
    int map1_;
    int map2_;
    int map3_;
};
