#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Blackdetect : public FilterBase {
public:
    /**
     * Detect video intervals that are (almost) black.
     */
    /**
     * set minimum detected black duration in seconds
     * Aliases: d
     * Type: Double
     * Required: No
     * Default: 2.00
     */
    void setBlack_min_duration(double value);
    double getBlack_min_duration() const;

    /**
     * set the picture black ratio threshold
     * Aliases: pic_th
     * Type: Double
     * Required: No
     * Default: 0.98
     */
    void setPicture_black_ratio_th(double value);
    double getPicture_black_ratio_th() const;

    /**
     * set the pixel black threshold
     * Aliases: pix_th
     * Type: Double
     * Required: No
     * Default: 0.10
     */
    void setPixel_black_th(double value);
    double getPixel_black_th() const;

    Blackdetect(double black_min_duration = 2.00, double picture_black_ratio_th = 0.98, double pixel_black_th = 0.10);
    virtual ~Blackdetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double black_min_duration_;
    double picture_black_ratio_th_;
    double pixel_black_th_;
};
