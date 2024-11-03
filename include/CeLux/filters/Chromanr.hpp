#pragma once
#include "FilterBase.hpp"
#include <string>
#include <vector>
#include <map>
#include <utility>

class Chromanr : public FilterBase {
public:
    /**
     * Reduce chrominance noise.
     */
    /**
     * set y+u+v threshold
     * Type: Float
     * Required: No
     * Default: 30.00
     */
    void setThres(float value);
    float getThres() const;

    /**
     * set horizontal patch size
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setSizew(int value);
    int getSizew() const;

    /**
     * set vertical patch size
     * Type: Integer
     * Required: No
     * Default: 5
     */
    void setSizeh(int value);
    int getSizeh() const;

    /**
     * set horizontal step
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setStepw(int value);
    int getStepw() const;

    /**
     * set vertical step
     * Type: Integer
     * Required: No
     * Default: 1
     */
    void setSteph(int value);
    int getSteph() const;

    /**
     * set y threshold
     * Type: Float
     * Required: No
     * Default: 200.00
     */
    void setThrey(float value);
    float getThrey() const;

    /**
     * set u threshold
     * Type: Float
     * Required: No
     * Default: 200.00
     */
    void setThreu(float value);
    float getThreu() const;

    /**
     * set v threshold
     * Type: Float
     * Required: No
     * Default: 200.00
     */
    void setThrev(float value);
    float getThrev() const;

    /**
     * set distance type
     * Unit: distance
     * Possible Values: manhattan (0), euclidean (1)
     * Type: Integer
     * Required: No
     * Default: 0
     */
    void setDistance(int value);
    int getDistance() const;

    Chromanr(float thres = 30.00, int sizew = 5, int sizeh = 5, int stepw = 1, int steph = 1, float threy = 200.00, float threu = 200.00, float threv = 200.00, int distance = 0);
    virtual ~Chromanr();

    std::string getFilterDescription() const override;

private:
    // Option variables
    float thres_;
    int sizew_;
    int sizeh_;
    int stepw_;
    int steph_;
    float threy_;
    float threu_;
    float threv_;
    int distance_;
};
