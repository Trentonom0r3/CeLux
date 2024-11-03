#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Volumedetect : public FilterBase {
public:
    /**
     * Detect audio volume.
     */
    Volumedetect();
    virtual ~Volumedetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
