#include "Estdif_bindings.hpp"

namespace py = pybind11;

void bind_Estdif(py::module_ &m) {
    py::class_<Estdif, FilterBase, std::shared_ptr<Estdif>>(m, "Estdif")
        .def(py::init<int, int, int, int, int, int, int, int, int>(),
             py::arg("mode") = 1,
             py::arg("parity") = -1,
             py::arg("deint") = 0,
             py::arg("rslope") = 1,
             py::arg("redge") = 2,
             py::arg("ecost") = 2,
             py::arg("mcost") = 1,
             py::arg("dcost") = 1,
             py::arg("interp") = 1)
        .def("setMode", &Estdif::setMode)
        .def("getMode", &Estdif::getMode)
        .def("setParity", &Estdif::setParity)
        .def("getParity", &Estdif::getParity)
        .def("setDeint", &Estdif::setDeint)
        .def("getDeint", &Estdif::getDeint)
        .def("setRslope", &Estdif::setRslope)
        .def("getRslope", &Estdif::getRslope)
        .def("setRedge", &Estdif::setRedge)
        .def("getRedge", &Estdif::getRedge)
        .def("setEcost", &Estdif::setEcost)
        .def("getEcost", &Estdif::getEcost)
        .def("setMcost", &Estdif::setMcost)
        .def("getMcost", &Estdif::getMcost)
        .def("setDcost", &Estdif::setDcost)
        .def("getDcost", &Estdif::getDcost)
        .def("setInterp", &Estdif::setInterp)
        .def("getInterp", &Estdif::getInterp)
        ;
}