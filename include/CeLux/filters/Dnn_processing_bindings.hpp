#pragma once
#include "Dnn_processing.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Dnn_processing(py::module_ &m);
