#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Overlay : public FilterBase {
public:
    /**
     * Overlay a video source on top of the input.
     */
    /**
     * set the x expression
     * Type: String
     * Required: No
     * Default: 0
     */
    void setX(const std::string& value);
    std::string getX() const;

    /**
     * set the y expression
     * Type: String
     * Required: No
     * Default: 0
     */
    void setY(const std::string& value);
    std::string getY() const;

    /**
     * Action to take when encountering EOF from secondary input 
     * Unit: eof_action
     * Possible Values: repeat (0), endall (1), pass (2)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setEof_action(int value);
    int getEof_action() const;

    /**
     * specify when to evaluate expressions
     * Unit: eval
     * Possible Values: init (0), frame (1)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setEval(int value);
    int getEval() const;

    /**
     * force termination when the shortest input terminates
     * Type: Boolean
     * Required: No
     * Default: false
     */
    void setShortest(bool value);
    bool getShortest() const;

    /**
     * set output format
     * Unit: format
     * Possible Values: yuv420 (0), yuv420p10 (1), yuv422 (2), yuv422p10 (3), yuv444 (4), yuv444p10 (5), rgb (6), gbrp (7), auto (8)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setFormat(int value);
    int getFormat() const;

    /**
     * repeat overlay of the last overlay frame
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setRepeatlast(bool value);
    bool getRepeatlast() const;

    /**
     * alpha format
     * Unit: alpha_format
     * Possible Values: straight (0), premultiplied (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setAlpha(int value);
    int getAlpha() const;

    Overlay(const std::string& x = "0", const std::string& y = "0", int eof_action = 0, int eval = 1, bool shortest = false, int format = 0, bool repeatlast = true, int alpha = 0);
    virtual ~Overlay();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::string x_;
    std::string y_;
    int eof_action_;
    int eval_;
    bool shortest_;
    int format_;
    bool repeatlast_;
    int alpha_;
};
