#include "Deshake.hpp"
#include <sstream>

Deshake::Deshake(int xForTheRectangularSearchArea, int yForTheRectangularSearchArea, int widthForTheRectangularSearchArea, int heightForTheRectangularSearchArea, int rx, int ry, int edge, int blocksize, int contrast, int search, const std::string& filename, bool opencl) {
    // Initialize member variables from parameters
    this->xForTheRectangularSearchArea_ = xForTheRectangularSearchArea;
    this->yForTheRectangularSearchArea_ = yForTheRectangularSearchArea;
    this->widthForTheRectangularSearchArea_ = widthForTheRectangularSearchArea;
    this->heightForTheRectangularSearchArea_ = heightForTheRectangularSearchArea;
    this->rx_ = rx;
    this->ry_ = ry;
    this->edge_ = edge;
    this->blocksize_ = blocksize;
    this->contrast_ = contrast;
    this->search_ = search;
    this->filename_ = filename;
    this->opencl_ = opencl;
}

Deshake::~Deshake() {
    // Destructor implementation (if needed)
}

void Deshake::setXForTheRectangularSearchArea(int value) {
    xForTheRectangularSearchArea_ = value;
}

int Deshake::getXForTheRectangularSearchArea() const {
    return xForTheRectangularSearchArea_;
}

void Deshake::setYForTheRectangularSearchArea(int value) {
    yForTheRectangularSearchArea_ = value;
}

int Deshake::getYForTheRectangularSearchArea() const {
    return yForTheRectangularSearchArea_;
}

void Deshake::setWidthForTheRectangularSearchArea(int value) {
    widthForTheRectangularSearchArea_ = value;
}

int Deshake::getWidthForTheRectangularSearchArea() const {
    return widthForTheRectangularSearchArea_;
}

void Deshake::setHeightForTheRectangularSearchArea(int value) {
    heightForTheRectangularSearchArea_ = value;
}

int Deshake::getHeightForTheRectangularSearchArea() const {
    return heightForTheRectangularSearchArea_;
}

void Deshake::setRx(int value) {
    rx_ = value;
}

int Deshake::getRx() const {
    return rx_;
}

void Deshake::setRy(int value) {
    ry_ = value;
}

int Deshake::getRy() const {
    return ry_;
}

void Deshake::setEdge(int value) {
    edge_ = value;
}

int Deshake::getEdge() const {
    return edge_;
}

void Deshake::setBlocksize(int value) {
    blocksize_ = value;
}

int Deshake::getBlocksize() const {
    return blocksize_;
}

void Deshake::setContrast(int value) {
    contrast_ = value;
}

int Deshake::getContrast() const {
    return contrast_;
}

void Deshake::setSearch(int value) {
    search_ = value;
}

int Deshake::getSearch() const {
    return search_;
}

void Deshake::setFilename(const std::string& value) {
    filename_ = value;
}

std::string Deshake::getFilename() const {
    return filename_;
}

void Deshake::setOpencl(bool value) {
    opencl_ = value;
}

bool Deshake::getOpencl() const {
    return opencl_;
}

std::string Deshake::getFilterDescription() const  {
    std::ostringstream desc;
    desc << "deshake";

    bool first = true;

    if (xForTheRectangularSearchArea_ != -1) {
        desc << (first ? "=" : ":") << "x=" << xForTheRectangularSearchArea_;
        first = false;
    }
    if (yForTheRectangularSearchArea_ != -1) {
        desc << (first ? "=" : ":") << "y=" << yForTheRectangularSearchArea_;
        first = false;
    }
    if (widthForTheRectangularSearchArea_ != -1) {
        desc << (first ? "=" : ":") << "w=" << widthForTheRectangularSearchArea_;
        first = false;
    }
    if (heightForTheRectangularSearchArea_ != -1) {
        desc << (first ? "=" : ":") << "h=" << heightForTheRectangularSearchArea_;
        first = false;
    }
    if (rx_ != 16) {
        desc << (first ? "=" : ":") << "rx=" << rx_;
        first = false;
    }
    if (ry_ != 16) {
        desc << (first ? "=" : ":") << "ry=" << ry_;
        first = false;
    }
    if (edge_ != 3) {
        desc << (first ? "=" : ":") << "edge=" << edge_;
        first = false;
    }
    if (blocksize_ != 8) {
        desc << (first ? "=" : ":") << "blocksize=" << blocksize_;
        first = false;
    }
    if (contrast_ != 125) {
        desc << (first ? "=" : ":") << "contrast=" << contrast_;
        first = false;
    }
    if (search_ != 0) {
        desc << (first ? "=" : ":") << "search=" << search_;
        first = false;
    }
    if (!filename_.empty()) {
        desc << (first ? "=" : ":") << "filename=" << filename_;
        first = false;
    }
    if (opencl_ != false) {
        desc << (first ? "=" : ":") << "opencl=" << (opencl_ ? "1" : "0");
        first = false;
    }

    return desc.str();
}
