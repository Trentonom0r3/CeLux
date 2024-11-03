#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tlut2 : public FilterBase {
public:
    /**
     * Compute and apply a lookup table from two successive frames.
     */
    /**
     * set component #0 expression
     * Type: String
     * Required: No
     * Default: x
     */
    void setC0(const std::string& value);
    std::string getC0() const;

    /**
     * set component #1 expression
     * Type: String
     * Required: No
     * Default: x
     */
    void setC1(const std::string& value);
    std::string getC1() const;

    /**
     * set component #2 expression
     * Type: String
     * Required: No
     * Default: x
     */
    void setC2(const std::string& value);
    std::string getC2() const;

    /**
     * set component #3 expression
     * Type: String
     * Required: No
     * Default: x
     */
    void setC3(const std::string& value);
    std::string getC3() const;

    Tlut2(const std::string& c0 = "x", const std::string& c1 = "x", const std::string& c2 = "x", const std::string& c3 = "x");
    virtual ~Tlut2();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string c0_;
    std::string c1_;
    std::string c2_;
    std::string c3_;
};
