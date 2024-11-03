#include "Bwdif_bindings.hpp"

namespace py = pybind11;

void bind_Bwdif(py::module_ &m) {
    py::class_<Bwdif, FilterBase, std::shared_ptr<Bwdif>>(m, "Bwdif")
        .def(py::init<int, int, int>(),
             py::arg("mode") = 1,
             py::arg("parity") = -1,
             py::arg("deint") = 0)
        .def("setMode", &Bwdif::setMode)
        .def("getMode", &Bwdif::getMode)
        .def("setParity", &Bwdif::setParity)
        .def("getParity", &Bwdif::getParity)
        .def("setDeint", &Bwdif::setDeint)
        .def("getDeint", &Bwdif::getDeint)
        ;
}