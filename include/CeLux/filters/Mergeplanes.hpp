#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Mergeplanes : public FilterBase {
public:
    /**
     * Merge planes.
     */
    /**
     * set output pixel format
     * Type: Pixel Format
     * Required: No
     * Default: yuva444p
     */
    void setFormat(const std::string& value);
    std::string getFormat() const;

    /**
     * set 1st input to output stream mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap0s(int value);
    int getMap0s() const;

    /**
     * set 1st input to output plane mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap0p(int value);
    int getMap0p() const;

    /**
     * set 2nd input to output stream mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap1s(int value);
    int getMap1s() const;

    /**
     * set 2nd input to output plane mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap1p(int value);
    int getMap1p() const;

    /**
     * set 3rd input to output stream mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap2s(int value);
    int getMap2s() const;

    /**
     * set 3rd input to output plane mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap2p(int value);
    int getMap2p() const;

    /**
     * set 4th input to output stream mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap3s(int value);
    int getMap3s() const;

    /**
     * set 4th input to output plane mapping
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMap3p(int value);
    int getMap3p() const;

    Mergeplanes(const std::string& format = "yuva444p", int map0s = 0, int map0p = 0, int map1s = 0, int map1p = 0, int map2s = 0, int map2p = 0, int map3s = 0, int map3p = 0);
    virtual ~Mergeplanes();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string format_;
    int map0s_;
    int map0p_;
    int map1s_;
    int map1p_;
    int map2s_;
    int map2p_;
    int map3s_;
    int map3p_;
};
