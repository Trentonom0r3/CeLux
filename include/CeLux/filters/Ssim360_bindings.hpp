#pragma once
#include "Ssim360.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Ssim360(py::module_ &m);
