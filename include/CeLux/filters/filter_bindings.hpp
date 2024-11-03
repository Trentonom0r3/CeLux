#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void register_filters(py::module_ &m);
