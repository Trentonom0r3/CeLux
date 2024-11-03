#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Xstack : public FilterBase {
public:
    /**
     * Stack video inputs into custom layout.
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
     * set custom layout
     * Type: String
     * Required: Yes
     * Default: No Default
     */
    void setLayout(const std::string& value);
    std::string getLayout() const;

    /**
     * set fixed size grid layout
     * Type: Image Size
     * Required: Yes
     * Default: No Default
     */
    void setGrid(const std::pair<int, int>& value);
    std::pair<int, int> getGrid() const;

    /**
     * force termination when the shortest input terminates
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setShortest(bool value);
    bool getShortest() const;

    /**
     * set the color for unused pixels
     * Type: String
     * Required: No
     * Default: none
     */
    void setFill(const std::string& value);
    std::string getFill() const;

    Xstack(int inputs = 2, const std::string& layout = "", std::pair<int, int> grid = std::make_pair<int, int>(0, 1), bool shortest = false, const std::string& fill = "none");
    virtual ~Xstack();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int inputs_;
    std::string layout_;
    std::pair<int, int> grid_;
    bool shortest_;
    std::string fill_;
};
