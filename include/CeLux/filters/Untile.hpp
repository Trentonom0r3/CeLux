#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Untile : public FilterBase {
public:
    /**
     * Untile a frame into a sequence of frames.
     */
    /**
     * set grid size
     * Type: Image Size
     * Required: No
     * Default: 6x5
     */
    void setLayout(const std::pair<int, int>& value);
    std::pair<int, int> getLayout() const;

    Untile(std::pair<int, int> layout = std::make_pair<int, int>(0, 1));
    virtual ~Untile();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> layout_;
};
