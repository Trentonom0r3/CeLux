#include "Field_bindings.hpp"

namespace py = pybind11;

void bind_Field(py::module_ &m) {
    py::class_<Field, FilterBase, std::shared_ptr<Field>>(m, "Field")
        .def(py::init<int>(),
             py::arg("type") = 0)
        .def("setType", &Field::setType)
        .def("getType", &Field::getType)
        ;
}