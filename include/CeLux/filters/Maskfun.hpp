#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Maskfun : public FilterBase {
public:
    /**
     * Create Mask.
     */
    /**
     * set low threshold
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setLow(int value);
    int getLow() const;

    /**
     * set high threshold
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setHigh(int value);
    int getHigh() const;

    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * set fill value
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFill(int value);
    int getFill() const;

    /**
     * set sum value
     * Type: Integer
     * Required: No
     * Default: 10
     */
    void setSum(int value);
    int getSum() const;

    Maskfun(int low = 10, int high = 10, int planes = 15, int fill = 0, int sum = 10);
    virtual ~Maskfun();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int low_;
    int high_;
    int planes_;
    int fill_;
    int sum_;
};
