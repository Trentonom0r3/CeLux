#pragma once
#include "Testsrc2.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Testsrc2(py::module_ &m);
