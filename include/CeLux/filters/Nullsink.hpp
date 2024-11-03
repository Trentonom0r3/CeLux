#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Nullsink : public FilterBase {
public:
    /**
     * Do absolutely nothing with the input video.
     */
    Nullsink();
    virtual ~Nullsink();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
