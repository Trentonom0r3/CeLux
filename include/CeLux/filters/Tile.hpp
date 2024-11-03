#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tile : public FilterBase {
public:
    /**
     * Tile several successive frames together.
     */
    /**
     * set grid size
     * Type: Image Size
     * Required: No
     * Default: 6x5
     */
    void setLayout(const std::pair<int, int>& value);
    std::pair<int, int> getLayout() const;

    /**
     * set maximum number of frame to render
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setNb_frames(int value);
    int getNb_frames() const;

    /**
     * set outer border margin in pixels
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setMargin(int value);
    int getMargin() const;

    /**
     * set inner border thickness in pixels
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setPadding(int value);
    int getPadding() const;

    /**
     * set the color of the unused area
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    /**
     * set how many frames to overlap for each render
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setOverlap(int value);
    int getOverlap() const;

    /**
     * set how many frames to initially pad
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setInit_padding(int value);
    int getInit_padding() const;

    Tile(std::pair<int, int> layout = std::make_pair<int, int>(0, 1), int nb_frames = 0, int margin = 0, int padding = 0, const std::string& color = "black", int overlap = 0, int init_padding = 0);
    virtual ~Tile();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> layout_;
    int nb_frames_;
    int margin_;
    int padding_;
    std::string color_;
    int overlap_;
    int init_padding_;
};
