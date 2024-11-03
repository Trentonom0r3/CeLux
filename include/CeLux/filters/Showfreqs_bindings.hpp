#pragma once
#include "Showfreqs.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Showfreqs(py::module_ &m);
