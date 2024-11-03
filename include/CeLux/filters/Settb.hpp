#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Settb : public FilterBase {
public:
    /**
     * Set timebase for the video output link.
     */
    /**
     * set expression determining the output timebase
     * Aliases: tb
     * Type: String
     * Required: No
     * Default: intb
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    Settb(const std::string& expr = "intb");
    virtual ~Settb();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string expr_;
};
