#include "Hwmap_bindings.hpp"

namespace py = pybind11;

void bind_Hwmap(py::module_ &m) {
    py::class_<Hwmap, FilterBase, std::shared_ptr<Hwmap>>(m, "Hwmap")
        .def(py::init<int, std::string, int>(),
             py::arg("mode") = 3,
             py::arg("derive_device") = "",
             py::arg("reverse") = 0)
        .def("setMode", &Hwmap::setMode)
        .def("getMode", &Hwmap::getMode)
        .def("setDerive_device", &Hwmap::setDerive_device)
        .def("getDerive_device", &Hwmap::getDerive_device)
        .def("setReverse", &Hwmap::setReverse)
        .def("getReverse", &Hwmap::getReverse)
        ;
}