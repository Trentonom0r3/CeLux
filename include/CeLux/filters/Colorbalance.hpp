#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorbalance : public FilterBase {
public:
    /**
     * Adjust the color balance.
     */
    /**
     * set red shadows
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRs(float value);
    float getRs() const;

    /**
     * set green shadows
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setGs(float value);
    float getGs() const;

    /**
     * set blue shadows
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBs(float value);
    float getBs() const;

    /**
     * set red midtones
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRm(float value);
    float getRm() const;

    /**
     * set green midtones
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setGm(float value);
    float getGm() const;

    /**
     * set blue midtones
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBm(float value);
    float getBm() const;

    /**
     * set red highlights
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setRh(float value);
    float getRh() const;

    /**
     * set green highlights
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setGh(float value);
    float getGh() const;

    /**
     * set blue highlights
     * Type: Float
     * Required: No
     * Default: 0.00
     */
    void setBh(float value);
    float getBh() const;

    /**
     * preserve lightness
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setPl(bool value);
    bool getPl() const;

    Colorbalance(float rs = 0.00, float gs = 0.00, float bs = 0.00, float rm = 0.00, float gm = 0.00, float bm = 0.00, float rh = 0.00, float gh = 0.00, float bh = 0.00, bool pl = false);
    virtual ~Colorbalance();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float rs_;
    float gs_;
    float bs_;
    float rm_;
    float gm_;
    float bm_;
    float rh_;
    float gh_;
    float bh_;
    bool pl_;
};
