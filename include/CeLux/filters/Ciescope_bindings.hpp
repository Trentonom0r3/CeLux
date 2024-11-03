#pragma once
#include "Ciescope.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Ciescope(py::module_ &m);
