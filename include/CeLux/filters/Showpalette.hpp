#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showpalette : public FilterBase {
public:
    /**
     * Display frame palette.
     */
    /**
     * set pixel box size
     * Type: Integer
     * Required: No
     * Default: 30
     */
    void setPixelBoxSize(int value);
    int getPixelBoxSize() const;

    Showpalette(int pixelBoxSize = 30);
    virtual ~Showpalette();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int pixelBoxSize_;
};
