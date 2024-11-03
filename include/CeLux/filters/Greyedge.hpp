#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Greyedge : public FilterBase {
public:
    /**
     * Estimates scene illumination by grey edge assumption.
     */
    /**
     * set differentiation order
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setDifford(int value);
    int getDifford() const;

    /**
     * set Minkowski norm
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setMinknorm(int value);
    int getMinknorm() const;

    /**
     * set sigma
     * Type: Double
     * Required: No
     * Default: 1.00
     */
    void setSigma(double value);
    double getSigma() const;

    Greyedge(int difford = 1, int minknorm = 1, double sigma = 1.00);
    virtual ~Greyedge();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int difford_;
    int minknorm_;
    double sigma_;
};
