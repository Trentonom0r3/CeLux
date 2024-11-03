#include "Rgbashift_bindings.hpp"

namespace py = pybind11;

void bind_Rgbashift(py::module_ &m) {
    py::class_<Rgbashift, FilterBase, std::shared_ptr<Rgbashift>>(m, "Rgbashift")
        .def(py::init<int, int, int, int, int, int, int, int, int>(),
             py::arg("rh") = 0,
             py::arg("rv") = 0,
             py::arg("gh") = 0,
             py::arg("gv") = 0,
             py::arg("bh") = 0,
             py::arg("bv") = 0,
             py::arg("ah") = 0,
             py::arg("av") = 0,
             py::arg("edge") = 0)
        .def("setRh", &Rgbashift::setRh)
        .def("getRh", &Rgbashift::getRh)
        .def("setRv", &Rgbashift::setRv)
        .def("getRv", &Rgbashift::getRv)
        .def("setGh", &Rgbashift::setGh)
        .def("getGh", &Rgbashift::getGh)
        .def("setGv", &Rgbashift::setGv)
        .def("getGv", &Rgbashift::getGv)
        .def("setBh", &Rgbashift::setBh)
        .def("getBh", &Rgbashift::getBh)
        .def("setBv", &Rgbashift::setBv)
        .def("getBv", &Rgbashift::getBv)
        .def("setAh", &Rgbashift::setAh)
        .def("getAh", &Rgbashift::getAh)
        .def("setAv", &Rgbashift::setAv)
        .def("getAv", &Rgbashift::getAv)
        .def("setEdge", &Rgbashift::setEdge)
        .def("getEdge", &Rgbashift::getEdge)
        ;
}