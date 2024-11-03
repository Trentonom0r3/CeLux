#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Showinfo : public FilterBase {
public:
    /**
     * Show textual information for each video frame.
     */
    /**
     * calculate checksums
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setChecksum(bool value);
    bool getChecksum() const;

    /**
     * try to print user data unregistered SEI as ascii character when possible
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setUdu_sei_as_ascii(bool value);
    bool getUdu_sei_as_ascii() const;

    Showinfo(bool checksum = true, bool udu_sei_as_ascii = false);
    virtual ~Showinfo();

    std::string getFilterDescription() const override;

private:
    // Option variables
    bool checksum_;
    bool udu_sei_as_ascii_;
};
