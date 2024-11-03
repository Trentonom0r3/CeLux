#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fsync : public FilterBase {
public:
    /**
     * Synchronize video frames from external source.
     */
    /**
     * set the file name to use for frame sync
     * Aliases: f
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFile(const std::string& value);
    std::string getFile() const;

    Fsync(const std::string& file = "");
    virtual ~Fsync();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string file_;
};
