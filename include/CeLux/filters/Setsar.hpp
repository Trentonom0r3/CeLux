#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setsar : public FilterBase {
public:
    /**
     * Set the pixel sample aspect ratio.
     */
    /**
     * set sample (pixel) aspect ratio
     * Aliases: r, sar
     * Type: String
     * Required: No
     * Default: 0
     */
    void setRatio(const std::string& value);
    std::string getRatio() const;

    /**
     * set max value for nominator or denominator in the ratio
     * Type: Integer
     * Required: No
     * Default: 100
     */
    void setMax(int value);
    int getMax() const;

    Setsar(const std::string& ratio = "0", int max = 100);
    virtual ~Setsar();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string ratio_;
    int max_;
};
