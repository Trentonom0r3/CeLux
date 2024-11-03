#pragma once
#include "Showcqt.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Showcqt(py::module_ &m);
