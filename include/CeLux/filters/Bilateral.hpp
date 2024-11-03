#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Bilateral : public FilterBase {
public:
    /**
     * Apply Bilateral filter.
     */
    /**
     * set spatial sigma
     * Type: Float
     * Required: No
     * Default: 0.10
     */
    void setSigmaS(float value);
    float getSigmaS() const;

    /**
     * set range sigma
     * Type: Float
     * Required: No
     * Default: 0.10
     */
    void setSigmaR(float value);
    float getSigmaR() const;

    /**
     * set planes to filter
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setPlanes(int value);
    int getPlanes() const;

    Bilateral(float sigmaS = 0.10, float sigmaR = 0.10, int planes = 1);
    virtual ~Bilateral();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float sigmaS_;
    float sigmaR_;
    int planes_;
};
