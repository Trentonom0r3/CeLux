#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hwdownload : public FilterBase {
public:
    /**
     * Download a hardware frame to a normal frame
     */
    Hwdownload();
    virtual ~Hwdownload();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
