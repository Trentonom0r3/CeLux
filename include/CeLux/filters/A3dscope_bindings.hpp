#pragma once
#include "A3dscope.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_A3dscope(py::module_ &m);
