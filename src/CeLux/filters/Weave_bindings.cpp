#include "Weave_bindings.hpp"

namespace py = pybind11;

void bind_Weave(py::module_ &m) {
    py::class_<Weave, FilterBase, std::shared_ptr<Weave>>(m, "Weave")
        .def(py::init<int>(),
             py::arg("first_field") = 0)
        .def("setFirst_field", &Weave::setFirst_field)
        .def("getFirst_field", &Weave::getFirst_field)
        ;
}