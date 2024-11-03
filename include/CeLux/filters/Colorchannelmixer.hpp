#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorchannelmixer : public FilterBase {
public:
    /**
     * Adjust colors by mixing color channels.
     */
    /**
     * set the red gain for the red channel
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setRr(double value);
    double getRr() const;

    /**
     * set the green gain for the red channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setRg(double value);
    double getRg() const;

    /**
     * set the blue gain for the red channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setRb(double value);
    double getRb() const;

    /**
     * set the alpha gain for the red channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setRa(double value);
    double getRa() const;

    /**
     * set the red gain for the green channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setGr(double value);
    double getGr() const;

    /**
     * set the green gain for the green channel
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setGg(double value);
    double getGg() const;

    /**
     * set the blue gain for the green channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setGb(double value);
    double getGb() const;

    /**
     * set the alpha gain for the green channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setGa(double value);
    double getGa() const;

    /**
     * set the red gain for the blue channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setBr(double value);
    double getBr() const;

    /**
     * set the green gain for the blue channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setBg(double value);
    double getBg() const;

    /**
     * set the blue gain for the blue channel
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setBb(double value);
    double getBb() const;

    /**
     * set the alpha gain for the blue channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setBa(double value);
    double getBa() const;

    /**
     * set the red gain for the alpha channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAr(double value);
    double getAr() const;

    /**
     * set the green gain for the alpha channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAg(double value);
    double getAg() const;

    /**
     * set the blue gain for the alpha channel
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAb(double value);
    double getAb() const;

    /**
     * set the alpha gain for the alpha channel
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setAa(double value);
    double getAa() const;

    /**
     * set the preserve color mode
     * Unit: preserve
     * Possible Values: none (0), lum (1), max (2), avg (3), sum (4), nrm (5), pwr (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPc(int value);
    int getPc() const;

    /**
     * set the preserve color amount
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setPa(double value);
    double getPa() const;

    Colorchannelmixer(double rr = 1.00, double rg = 0.00, double rb = 0.00, double ra = 0.00, double gr = 0.00, double gg = 1.00, double gb = 0.00, double ga = 0.00, double br = 0.00, double bg = 0.00, double bb = 1.00, double ba = 0.00, double ar = 0.00, double ag = 0.00, double ab = 0.00, double aa = 1.00, int pc = 0, double pa = 0.00);
    virtual ~Colorchannelmixer();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double rr_;
    double rg_;
    double rb_;
    double ra_;
    double gr_;
    double gg_;
    double gb_;
    double ga_;
    double br_;
    double bg_;
    double bb_;
    double ba_;
    double ar_;
    double ag_;
    double ab_;
    double aa_;
    int pc_;
    double pa_;
};
