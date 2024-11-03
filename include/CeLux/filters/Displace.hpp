#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Displace : public FilterBase {
public:
    /**
     * Displace pixels.
     */
    /**
     * set edge mode
     * Unit: edge
     * Possible Values: blank (0), smear (1), wrap (2), mirror (3)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setEdge(int value);
    int getEdge() const;

    Displace(int edge = 1);
    virtual ~Displace();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int edge_;
};
