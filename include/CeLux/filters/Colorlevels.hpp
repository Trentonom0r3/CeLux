#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Colorlevels : public FilterBase {
public:
    /**
     * Adjust the color levels.
     */
    /**
     * set input red black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setRimin(double value);
    double getRimin() const;

    /**
     * set input green black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setGimin(double value);
    double getGimin() const;

    /**
     * set input blue black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setBimin(double value);
    double getBimin() const;

    /**
     * set input alpha black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAimin(double value);
    double getAimin() const;

    /**
     * set input red white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setRimax(double value);
    double getRimax() const;

    /**
     * set input green white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setGimax(double value);
    double getGimax() const;

    /**
     * set input blue white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setBimax(double value);
    double getBimax() const;

    /**
     * set input alpha white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setAimax(double value);
    double getAimax() const;

    /**
     * set output red black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setRomin(double value);
    double getRomin() const;

    /**
     * set output green black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setGomin(double value);
    double getGomin() const;

    /**
     * set output blue black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setBomin(double value);
    double getBomin() const;

    /**
     * set output alpha black point
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setAomin(double value);
    double getAomin() const;

    /**
     * set output red white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setRomax(double value);
    double getRomax() const;

    /**
     * set output green white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setGomax(double value);
    double getGomax() const;

    /**
     * set output blue white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setBomax(double value);
    double getBomax() const;

    /**
     * set output alpha white point
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setAomax(double value);
    double getAomax() const;

    /**
     * set preserve color mode
     * Unit: preserve
     * Possible Values: none (0), lum (1), max (2), avg (3), sum (4), nrm (5), pwr (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPreserve(int value);
    int getPreserve() const;

    Colorlevels(double rimin = 0.00, double gimin = 0.00, double bimin = 0.00, double aimin = 0.00, double rimax = 1.00, double gimax = 1.00, double bimax = 1.00, double aimax = 1.00, double romin = 0.00, double gomin = 0.00, double bomin = 0.00, double aomin = 0.00, double romax = 1.00, double gomax = 1.00, double bomax = 1.00, double aomax = 1.00, int preserve = 0);
    virtual ~Colorlevels();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double rimin_;
    double gimin_;
    double bimin_;
    double aimin_;
    double rimax_;
    double gimax_;
    double bimax_;
    double aimax_;
    double romin_;
    double gomin_;
    double bomin_;
    double aomin_;
    double romax_;
    double gomax_;
    double bomax_;
    double aomax_;
    int preserve_;
};
