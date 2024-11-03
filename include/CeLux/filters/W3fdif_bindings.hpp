#pragma once
#include "W3fdif.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_W3fdif(py::module_ &m);
