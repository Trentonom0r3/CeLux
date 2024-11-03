#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Pixdesctest : public FilterBase {
public:
    /**
     * Test pixel format definitions.
     */
    Pixdesctest();
    virtual ~Pixdesctest();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
