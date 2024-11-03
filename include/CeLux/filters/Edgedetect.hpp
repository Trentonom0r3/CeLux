#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Edgedetect : public FilterBase {
public:
    /**
     * Detect and draw edge.
     */
    /**
     * set high threshold
     * Type: Double
     * Required: No
     * Default: 0.20
     */
    void setHigh(double value);
    double getHigh() const;

    /**
     * set low threshold
     * Type: Double
     * Required: No
     * Default: 0.08
     */
    void setLow(double value);
    double getLow() const;

    /**
     * set mode
     * Unit: mode
     * Possible Values: wires (0), colormix (1), canny (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set planes to filter
     * Unit: flags
     * Possible Values: y (1), u (2), v (4), r (4), g (1), b (2)
     * Type: Flags
     * Required: No
     * Default: 7
     */
    void setPlanes(int value);
    int getPlanes() const;

    Edgedetect(double high = 0.20, double low = 0.08, int mode = 0, int planes = 7);
    virtual ~Edgedetect();

    std::string getFilterDescription() const override;

private:
    // Option variables
    double high_;
    double low_;
    int mode_;
    int planes_;
};
