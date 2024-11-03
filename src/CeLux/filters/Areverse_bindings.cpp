#include "Areverse_bindings.hpp"

namespace py = pybind11;

void bind_Areverse(py::module_ &m) {
    py::class_<Areverse, FilterBase, std::shared_ptr<Areverse>>(m, "Areverse")
        .def(py::init<>())
        ;
}