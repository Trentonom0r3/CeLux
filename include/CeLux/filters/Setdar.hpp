#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setdar : public FilterBase {
public:
    /**
     * Set the frame display aspect ratio.
     */
    /**
     * set display aspect ratio
     * Aliases: r, dar
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

    Setdar(const std::string& ratio = "0", int max = 100);
    virtual ~Setdar();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string ratio_;
    int max_;
};
