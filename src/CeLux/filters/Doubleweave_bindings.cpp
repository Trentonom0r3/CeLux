#include "Doubleweave_bindings.hpp"

namespace py = pybind11;

void bind_Doubleweave(py::module_ &m) {
    py::class_<Doubleweave, FilterBase, std::shared_ptr<Doubleweave>>(m, "Doubleweave")
        .def(py::init<int>(),
             py::arg("first_field") = 0)
        .def("setFirst_field", &Doubleweave::setFirst_field)
        .def("getFirst_field", &Doubleweave::getFirst_field)
        ;
}