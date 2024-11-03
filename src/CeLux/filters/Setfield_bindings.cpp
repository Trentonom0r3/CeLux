#include "Setfield_bindings.hpp"

namespace py = pybind11;

void bind_Setfield(py::module_ &m) {
    py::class_<Setfield, FilterBase, std::shared_ptr<Setfield>>(m, "Setfield")
        .def(py::init<int>(),
             py::arg("mode") = -1)
        .def("setMode", &Setfield::setMode)
        .def("getMode", &Setfield::getMode)
        ;
}