#include "Chromashift_bindings.hpp"

namespace py = pybind11;

void bind_Chromashift(py::module_ &m) {
    py::class_<Chromashift, FilterBase, std::shared_ptr<Chromashift>>(m, "Chromashift")
        .def(py::init<int, int, int, int, int>(),
             py::arg("cbh") = 0,
             py::arg("cbv") = 0,
             py::arg("crh") = 0,
             py::arg("crv") = 0,
             py::arg("edge") = 0)
        .def("setCbh", &Chromashift::setCbh)
        .def("getCbh", &Chromashift::getCbh)
        .def("setCbv", &Chromashift::setCbv)
        .def("getCbv", &Chromashift::getCbv)
        .def("setCrh", &Chromashift::setCrh)
        .def("getCrh", &Chromashift::getCrh)
        .def("setCrv", &Chromashift::setCrv)
        .def("getCrv", &Chromashift::getCrv)
        .def("setEdge", &Chromashift::setEdge)
        .def("getEdge", &Chromashift::getEdge)
        ;
}