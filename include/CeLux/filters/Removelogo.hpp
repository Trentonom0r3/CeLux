#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Removelogo : public FilterBase {
public:
    /**
     * Remove a TV logo based on a mask image.
     */
    /**
     * set bitmap filename
     * Aliases: f
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFilename(const std::string& value);
    std::string getFilename() const;

    Removelogo(const std::string& filename = "");
    virtual ~Removelogo();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string filename_;
};
