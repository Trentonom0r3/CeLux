#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Earwax : public FilterBase {
public:
    /**
     * Widen the stereo image.
     */
    Earwax();
    virtual ~Earwax();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
