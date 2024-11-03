#include "Amultiply_bindings.hpp"

namespace py = pybind11;

void bind_Amultiply(py::module_ &m) {
    py::class_<Amultiply, FilterBase, std::shared_ptr<Amultiply>>(m, "Amultiply")
        .def(py::init<>())
        ;
}