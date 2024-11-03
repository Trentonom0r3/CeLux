#include "Deflicker_bindings.hpp"

namespace py = pybind11;

void bind_Deflicker(py::module_ &m) {
    py::class_<Deflicker, FilterBase, std::shared_ptr<Deflicker>>(m, "Deflicker")
        .def(py::init<int, int, bool>(),
             py::arg("size") = 5,
             py::arg("mode") = 0,
             py::arg("bypass") = false)
        .def("setSize", &Deflicker::setSize)
        .def("getSize", &Deflicker::getSize)
        .def("setMode", &Deflicker::setMode)
        .def("getMode", &Deflicker::getMode)
        .def("setBypass", &Deflicker::setBypass)
        .def("getBypass", &Deflicker::getBypass)
        ;
}