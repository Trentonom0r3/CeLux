#pragma once
#include "Dnn_classify.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Dnn_classify(py::module_ &m);
