#include "Swapuv_bindings.hpp"

namespace py = pybind11;

void bind_Swapuv(py::module_ &m) {
    py::class_<Swapuv, FilterBase, std::shared_ptr<Swapuv>>(m, "Swapuv")
        .def(py::init<>())
        ;
}