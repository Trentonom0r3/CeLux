#pragma once
#include "Allrgb.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Allrgb(py::module_ &m);
