#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hflip : public FilterBase {
public:
    /**
     * Horizontally flip the input video.
     */
    Hflip();
    virtual ~Hflip();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
