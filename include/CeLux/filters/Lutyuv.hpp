#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Lutyuv : public FilterBase {
public:
    /**
     * Compute and apply a lookup table to the YUV input video.
     */
    /**
     * set component #0 expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setC0(const std::string& value);
    std::string getC0() const;

    /**
     * set component #1 expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setC1(const std::string& value);
    std::string getC1() const;

    /**
     * set component #2 expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setC2(const std::string& value);
    std::string getC2() const;

    /**
     * set component #3 expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setC3(const std::string& value);
    std::string getC3() const;

    /**
     * set Y expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setY(const std::string& value);
    std::string getY() const;

    /**
     * set U expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setU(const std::string& value);
    std::string getU() const;

    /**
     * set V expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setV(const std::string& value);
    std::string getV() const;

    /**
     * set R expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setR(const std::string& value);
    std::string getR() const;

    /**
     * set G expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setG(const std::string& value);
    std::string getG() const;

    /**
     * set B expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setB(const std::string& value);
    std::string getB() const;

    /**
     * set A expression
     * Type: String
     * Required: No
     * Default: clipval
     */
    void setA(const std::string& value);
    std::string getA() const;

    Lutyuv(const std::string& c0 = "clipval", const std::string& c1 = "clipval", const std::string& c2 = "clipval", const std::string& c3 = "clipval", const std::string& y = "clipval", const std::string& u = "clipval", const std::string& v = "clipval", const std::string& r = "clipval", const std::string& g = "clipval", const std::string& b = "clipval", const std::string& a = "clipval");
    virtual ~Lutyuv();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string c0_;
    std::string c1_;
    std::string c2_;
    std::string c3_;
    std::string y_;
    std::string u_;
    std::string v_;
    std::string r_;
    std::string g_;
    std::string b_;
    std::string a_;
};
