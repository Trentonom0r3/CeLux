#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Hstack : public FilterBase {
public:
    /**
     * Stack video inputs horizontally.
     */
    /**
     * set number of inputs
     * Type: Integer
     * Required: No
     * Default: 2
     */
    void setInputs(int value);
    int getInputs() const;

    /**
     * force termination when the shortest input terminates
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setShortest(bool value);
    bool getShortest() const;

    Hstack(int inputs = 2, bool shortest = false);
    virtual ~Hstack();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int inputs_;
    bool shortest_;
};
