#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Transpose : public FilterBase {
public:
    /**
     * Transpose input video.
     */
    /**
     * set transpose direction
     * Unit: dir
     * Possible Values: cclock_flip (0), clock (1), cclock (2), clock_flip (3)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDir(int value);
    int getDir() const;

    /**
     * do not apply transposition if the input matches the specified geometry
     * Unit: passthrough
     * Possible Values: none (0), portrait (2), landscape (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPassthrough(int value);
    int getPassthrough() const;

    Transpose(int dir = 0, int passthrough = 0);
    virtual ~Transpose();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int dir_;
    int passthrough_;
};
