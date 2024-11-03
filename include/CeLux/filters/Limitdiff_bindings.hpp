#pragma once
#include "Limitdiff.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Limitdiff(py::module_ &m);
