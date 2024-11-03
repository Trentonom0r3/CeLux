#pragma once
#include "Thistogram.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Thistogram(py::module_ &m);
