#pragma once
#include "Readeia608.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Readeia608(py::module_ &m);
