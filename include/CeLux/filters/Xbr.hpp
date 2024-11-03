#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Xbr : public FilterBase {
public:
    /**
     * Scale the input using xBR algorithm.
     */
    /**
     * set scale factor
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setScaleFactor(int value);
    int getScaleFactor() const;

    Xbr(int scaleFactor = 3);
    virtual ~Xbr();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int scaleFactor_;
};
