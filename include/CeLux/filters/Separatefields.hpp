#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Separatefields : public FilterBase {
public:
    /**
     * Split input video frames into fields.
     */
    Separatefields();
    virtual ~Separatefields();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
