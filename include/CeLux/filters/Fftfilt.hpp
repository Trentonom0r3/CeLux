#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fftfilt : public FilterBase {
public:
    /**
     * Apply arbitrary expressions to pixels in frequency domain.
     */
    /**
     * adjust gain in Y plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDc_Y(int value);
    int getDc_Y() const;

    /**
     * adjust gain in U plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDc_U(int value);
    int getDc_U() const;

    /**
     * adjust gain in V plane
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDc_V(int value);
    int getDc_V() const;

    /**
     * set luminance expression in Y plane
     * Type: String
     * Required: No
     * Default: 1
     */
    void setWeight_Y(const std::string& value);
    std::string getWeight_Y() const;

    /**
     * set chrominance expression in U plane
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setWeight_U(const std::string& value);
    std::string getWeight_U() const;

    /**
     * set chrominance expression in V plane
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setWeight_V(const std::string& value);
    std::string getWeight_V() const;

    /**
     * specify when to evaluate expressions
     * Unit: eval
     * Possible Values: init (0), frame (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEval(int value);
    int getEval() const;

    Fftfilt(int dc_Y = 0, int dc_U = 0, int dc_V = 0, const std::string& weight_Y = "1", const std::string& weight_U = "", const std::string& weight_V = "", int eval = 0);
    virtual ~Fftfilt();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dc_Y_;
    int dc_U_;
    int dc_V_;
    std::string weight_Y_;
    std::string weight_U_;
    std::string weight_V_;
    int eval_;
};
