#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Grayworld : public FilterBase {
public:
    /**
     * Adjust white balance using LAB gray world algorithm
     */
    Grayworld();
    virtual ~Grayworld();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
