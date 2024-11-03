#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Ccrepack : public FilterBase {
public:
    /**
     * Repack CEA-708 closed caption metadata
     */
    Ccrepack();
    virtual ~Ccrepack();

    std::string getFilterDescription() const override;

private:
    // Option variables
};
