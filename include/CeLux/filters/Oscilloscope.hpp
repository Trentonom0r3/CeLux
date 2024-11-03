#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Oscilloscope : public FilterBase {
public:
    /**
     * 2D Video Oscilloscope.
     */
    /**
     * set scope x position
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setScopeXPosition(float value);
    float getScopeXPosition() const;

    /**
     * set scope y position
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setScopeYPosition(float value);
    float getScopeYPosition() const;

    /**
     * set scope size
     * Type: Float
     * Required: No
     * Default: 0.80
     */
    void setScopeSize(float value);
    float getScopeSize() const;

    /**
     * set scope tilt
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setScopeTilt(float value);
    float getScopeTilt() const;

    /**
     * set trace opacity
     * Type: Float
     * Required: No
     * Default: 0.80
     */
    void setTraceOpacity(float value);
    float getTraceOpacity() const;

    /**
     * set trace x position
     * Type: Float
     * Required: No
     * Default: 0.50
     */
    void setTx(float value);
    float getTx() const;

    /**
     * set trace y position
     * Type: Float
     * Required: No
     * Default: 0.90
     */
    void setTy(float value);
    float getTy() const;

    /**
     * set trace width
     * Type: Float
     * Required: No
     * Default: 0.80
     */
    void setTw(float value);
    float getTw() const;

    /**
     * set trace height
     * Type: Float
     * Required: No
     * Default: 0.30
     */
    void setTh(float value);
    float getTh() const;

    /**
     * set components to trace
     * Type: Integer
     * Required: No
     * Default: 7
     */
    void setComponentsToTrace(int value);
    int getComponentsToTrace() const;

    /**
     * draw trace grid
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setDrawTraceGrid(bool value);
    bool getDrawTraceGrid() const;

    /**
     * draw statistics
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setSt(bool value);
    bool getSt() const;

    /**
     * draw scope
     * Type: Boolean
     * Required: No
     * Default: true
     */
    void setSc(bool value);
    bool getSc() const;

    Oscilloscope(float scopeXPosition = 0.50, float scopeYPosition = 0.50, float scopeSize = 0.80, float scopeTilt = 0.50, float traceOpacity = 0.80, float tx = 0.50, float ty = 0.90, float tw = 0.80, float th = 0.30, int componentsToTrace = 7, bool drawTraceGrid = true, bool st = true, bool sc = true);
    virtual ~Oscilloscope();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float scopeXPosition_;
    float scopeYPosition_;
    float scopeSize_;
    float scopeTilt_;
    float traceOpacity_;
    float tx_;
    float ty_;
    float tw_;
    float th_;
    int componentsToTrace_;
    bool drawTraceGrid_;
    bool st_;
    bool sc_;
};
