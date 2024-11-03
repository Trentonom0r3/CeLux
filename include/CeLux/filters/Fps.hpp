#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Fps : public FilterBase {
public:
    /**
     * Force constant framerate.
     */
    /**
     * A string describing desired output framerate
     * Type: String
     * Required: No
     * Default: 25
     */
    void setFps(const std::string& value);
    std::string getFps() const;

    /**
     * Assume the first PTS should be this value.
     * Type: Double
     * Required: No
     * Default: 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00
     */
    void setStart_time(double value);
    double getStart_time() const;

    /**
     * set rounding method for timestamps
     * Unit: round
     * Possible Values: zero (0), inf (1), down (2), up (3), near (5)
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setRound(int value);
    int getRound() const;

    /**
     * action performed for last frame
     * Unit: eof_action
     * Possible Values: round (0), pass (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEof_action(int value);
    int getEof_action() const;

    Fps(const std::string& fps = "25", double start_time = 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00, int round = 5, int eof_action = 0);
    virtual ~Fps();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string fps_;
    double start_time_;
    int round_;
    int eof_action_;
};
