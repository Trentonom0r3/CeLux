#pragma once
#include "Hwupload_cuda.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "FilterFactory.hpp"

namespace py = pybind11;

void bind_Hwupload_cuda(py::module_ &m);
