#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Metadata : public FilterBase {
public:
    /**
     * Manipulate video frame metadata.
     */
    /**
     * set a mode of operation
     * Unit: mode
     * Possible Values: select (0), add (1), modify (2), delete (3), print (4)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMode(int value);
    int getMode() const;

    /**
     * set metadata key
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setKey(const std::string& value);
    std::string getKey() const;

    /**
     * set metadata value
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setValue(const std::string& value);
    std::string getValue() const;

    /**
     * function for comparing values
     * Unit: function
     * Possible Values: same_str (0), starts_with (1), less (2), equal (3), greater (4), expr (5), ends_with (6)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFunction(int value);
    int getFunction() const;

    /**
     * set expression for expr function
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setExpr(const std::string& value);
    std::string getExpr() const;

    /**
     * set file where to print metadata information
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setFile(const std::string& value);
    std::string getFile() const;

    /**
     * reduce buffering when printing to user-set file or pipe
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setDirect(bool value);
    bool getDirect() const;

    Metadata(int mode = 0, const std::string& key = "", const std::string& value = "", int function = 0, const std::string& expr = "", const std::string& file = "", bool direct = false);
    virtual ~Metadata();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int mode_;
    std::string key_;
    std::string value_;
    int function_;
    std::string expr_;
    std::string file_;
    bool direct_;
};
