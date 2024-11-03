#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Alphaextract : public FilterBase {
public:
    /**
     * Extract an alpha channel as a grayscale image component.
     */
    Alphaextract();
    virtual ~Alphaextract();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
