#pragma once
#include "Pal100bars.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Pal100bars(py::module_ &m);
