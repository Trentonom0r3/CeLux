#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Unpremultiply : public FilterBase {
public:
    /**
     * UnPreMultiply first stream with first plane of second stream.
     */
    /**
     * set planes
     * Type: Integer
     * Required: No
     * Default: 15
     */
    void setPlanes(int value);
    int getPlanes() const;

    /**
     * enable inplace mode
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setInplace(bool value);
    bool getInplace() const;

    Unpremultiply(int planes = 15, bool inplace = false);
    virtual ~Unpremultiply();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int planes_;
    bool inplace_;
};
