#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Mandelbrot : public FilterBase {
public:
    /**
     * Render a Mandelbrot fractal.
     */
    /**
     * set frame size
     * Aliases: s
     * Type: Image Size
     * Required: No
     * Default: 640x480
     */
    void setSize(const std::pair<int, int>& value);
    std::pair<int, int> getSize() const;

    /**
     * set frame rate
     * Aliases: r
     * Type: Video Rate
     * Required: No
     * Default: 40439.8
     */
    void setRate(const std::pair<int, int>& value);
    std::pair<int, int> getRate() const;

    /**
     * set max iterations number
     * Type: Integer
     * Required: No
     * Default: 7189
     */
    void setMaxiter(int value);
    int getMaxiter() const;

    /**
     * set the initial x position
     * Type: Double
     * Required: No
     * Default: -0.74
     */
    void setStart_x(double value);
    double getStart_x() const;

    /**
     * set the initial y position
     * Type: Double
     * Required: No
     * Default: -0.13
     */
    void setStart_y(double value);
    double getStart_y() const;

    /**
     * set the initial scale value
     * Type: Double
     * Required: No
     * Default: 3.00
     */
    void setStart_scale(double value);
    double getStart_scale() const;

    /**
     * set the terminal scale value
     * Type: Double
     * Required: No
     * Default: 0.30
     */
    void setEnd_scale(double value);
    double getEnd_scale() const;

    /**
     * set the terminal pts value
     * Type: Double
     * Required: No
     * Default: 400.00
     */
    void setEnd_pts(double value);
    double getEnd_pts() const;

    /**
     * set the bailout value
     * Type: Double
     * Required: No
     * Default: 10.00
     */
    void setBailout(double value);
    double getBailout() const;

    /**
     * set morph x frequency
     * Type: Double
     * Required: No
     * Default: 0.01
     */
    void setMorphxf(double value);
    double getMorphxf() const;

    /**
     * set morph y frequency
     * Type: Double
     * Required: No
     * Default: 0.01
     */
    void setMorphyf(double value);
    double getMorphyf() const;

    /**
     * set morph amplitude
     * Type: Double
     * Required: No
     * Default: 0.00
     */
    void setMorphamp(double value);
    double getMorphamp() const;

    /**
     * set outer coloring mode
     * Unit: outer
     * Possible Values: iteration_count (0), normalized_iteration_count (1), white (2), outz (3)
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setOuter(int value);
    int getOuter() const;

    /**
     * set inner coloring mode
     * Unit: inner
     * Possible Values: black (0), period (1), convergence (2), mincol (3)
     * Type: Integer
     * Required: No
     * Default: 3
     */
    void setInner(int value);
    int getInner() const;

    Mandelbrot(std::pair<int, int> size = std::make_pair<int, int>(0, 1), std::pair<int, int> rate = std::make_pair<int, int>(0, 1), int maxiter = 7189, double start_x = -0.74, double start_y = -0.13, double start_scale = 3.00, double end_scale = 0.30, double end_pts = 400.00, double bailout = 10.00, double morphxf = 0.01, double morphyf = 0.01, double morphamp = 0.00, int outer = 1, int inner = 3);
    virtual ~Mandelbrot();

    std::string getFilterDescription() const override;

private:
    // Option variables
    std::pair<int, int> size_;
    std::pair<int, int> rate_;
    int maxiter_;
    double start_x_;
    double start_y_;
    double start_scale_;
    double end_scale_;
    double end_pts_;
    double bailout_;
    double morphxf_;
    double morphyf_;
    double morphamp_;
    int outer_;
    int inner_;
};
