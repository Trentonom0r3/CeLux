#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Setpts : public FilterBase {
public:
    /**
     * Set PTS for the output video frame.
     */
    /**
     * Expression determining the frame timestamp
     * Type: String
     * Required: No
     * Default: PTS
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    Setpts(const std::string& expr = "PTS");
    virtual ~Setpts();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string expr_;
};
