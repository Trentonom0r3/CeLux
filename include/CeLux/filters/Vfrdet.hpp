#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vfrdet : public FilterBase {
public:
    /**
     * Variable frame rate detect filter.
     */
    Vfrdet();
    virtual ~Vfrdet();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
