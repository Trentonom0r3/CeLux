#pragma once
#include <algorithm>
#include <cstdint> // For fixed-width integer types
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>   // For std::ostream
#include <stdexcept> // For std::runtime_error
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class FilterBase {
public:
    FilterBase();
    virtual ~FilterBase();

    /**
     * Get a description of the filter and its options.
     * This function should be overridden by subclasses.
     */
    virtual std::string getFilterDescription() const;

protected:
    // Shared protected members (if any)

private:
    // Shared private members (if any)
};
