#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Qp : public FilterBase {
public:
    /**
     * Change video quantization parameters.
     */
    /**
     * set qp expression
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setQp(const std::string& value);
    std::string getQp() const;

    Qp(const std::string& qp = "");
    virtual ~Qp();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string qp_;
};
