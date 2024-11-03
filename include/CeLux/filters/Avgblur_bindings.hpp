#pragma once
#include "Avgblur.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Avgblur(py::module_ &m);
