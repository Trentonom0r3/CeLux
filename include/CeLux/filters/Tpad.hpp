#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Tpad : public FilterBase {
public:
    /**
     * Temporarily pad video frames.
     */
    /**
     * set the number of frames to delay input
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStart(int value);
    int getStart() const;

    /**
     * set the number of frames to add after input finished
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStop(int value);
    int getStop() const;

    /**
     * set the mode of added frames to start
     * Unit: mode
     * Possible Values: add (0), clone (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStart_mode(int value);
    int getStart_mode() const;

    /**
     * set the mode of added frames to end
     * Unit: mode
     * Possible Values: add (0), clone (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setStop_mode(int value);
    int getStop_mode() const;

    /**
     * set the duration to delay input
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setStart_duration(int64_t value);
    int64_t getStart_duration() const;

    /**
     * set the duration to pad input
     * Type: Duration
     * Required: No
     * Default: 0
     */
    void setStop_duration(int64_t value);
    int64_t getStop_duration() const;

    /**
     * set the color of the added frames
     * Type: Color
     * Required: No
     * Default: black
     */
    void setColor(const std::string& value);
    std::string getColor() const;

    Tpad(int start = 0, int stop = 0, int start_mode = 0, int stop_mode = 0, int64_t start_duration = 0ULL, int64_t stop_duration = 0ULL, const std::string& color = "black");
    virtual ~Tpad();

    std::string getFilterDescription() const override;

private:
    // Option variables
    int start_;
    int stop_;
    int start_mode_;
    int stop_mode_;
    int64_t start_duration_;
    int64_t stop_duration_;
    std::string color_;
};
