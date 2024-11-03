#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Vibrance : public FilterBase {
public:
    /**
     * Boost or alter saturation.
     */
    /**
     * set the intensity value
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setIntensity(float value);
    float getIntensity() const;

    /**
     * set the red balance value
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setRbal(float value);
    float getRbal() const;

    /**
     * set the green balance value
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setGbal(float value);
    float getGbal() const;

    /**
     * set the blue balance value
     * Type: Float
     * Required: No
     * Default: 1.00
     */
    void setBbal(float value);
    float getBbal() const;

    /**
     * set the red luma coefficient
     * Type: Float
     * Required: No
     * Default: 0.07
     */
    void setRlum(float value);
    float getRlum() const;

    /**
     * set the green luma coefficient
     * Type: Float
     * Required: No
     * Default: 0.72
     */
    void setGlum(float value);
    float getGlum() const;

    /**
     * set the blue luma coefficient
     * Type: Float
     * Required: No
     * Default: 0.21
     */
    void setBlum(float value);
    float getBlum() const;

    /**
     * use alternate colors
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setAlternate(bool value);
    bool getAlternate() const;

    Vibrance(float intensity = 0.00, float rbal = 1.00, float gbal = 1.00, float bbal = 1.00, float rlum = 0.07, float glum = 0.72, float blum = 0.21, bool alternate = false);
    virtual ~Vibrance();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float intensity_;
    float rbal_;
    float gbal_;
    float bbal_;
    float rlum_;
    float glum_;
    float blum_;
    bool alternate_;
};
