#include "Telecine_bindings.hpp"

namespace py = pybind11;

void bind_Telecine(py::module_ &m) {
    py::class_<Telecine, FilterBase, std::shared_ptr<Telecine>>(m, "Telecine")
        .def(py::init<int, std::string>(),
             py::arg("first_field") = 0,
             py::arg("pattern") = "23")
        .def("setFirst_field", &Telecine::setFirst_field)
        .def("getFirst_field", &Telecine::getFirst_field)
        .def("setPattern", &Telecine::setPattern)
        .def("getPattern", &Telecine::getPattern)
        ;
}