#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Nlmeans : public FilterBase {
public:
    /**
     * Non-local means denoiser.
     */
    /**
     * denoising strength
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setDenoisingStrength(double value);
    double getDenoisingStrength() const;

    /**
     * patch size
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setPatchSize(int value);
    int getPatchSize() const;

    /**
     * patch size for chroma planes
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPc(int value);
    int getPc() const;

    /**
     * research window
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setResearchWindow(int value);
    int getResearchWindow() const;

    /**
     * research window for chroma planes
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setRc(int value);
    int getRc() const;

    Nlmeans(double denoisingStrength = 1.00, int patchSize = 7, int pc = 0, int researchWindow = 15, int rc = 0);
    virtual ~Nlmeans();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double denoisingStrength_;
    int patchSize_;
    int pc_;
    int researchWindow_;
    int rc_;
};
