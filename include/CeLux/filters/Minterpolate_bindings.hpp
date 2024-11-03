#pragma once
#include "Minterpolate.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Minterpolate(py::module_ &m);
