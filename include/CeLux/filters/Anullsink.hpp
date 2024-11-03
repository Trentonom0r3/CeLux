#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Anullsink : public FilterBase {
public:
    /**
     * Do absolutely nothing with the input audio.
     */
    Anullsink();
    virtual ~Anullsink();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
