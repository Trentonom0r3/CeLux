#include "Aderivative_bindings.hpp"

namespace py = pybind11;

void bind_Aderivative(py::module_ &m) {
    py::class_<Aderivative, FilterBase, std::shared_ptr<Aderivative>>(m, "Aderivative")
        .def(py::init<>())
        ;
}