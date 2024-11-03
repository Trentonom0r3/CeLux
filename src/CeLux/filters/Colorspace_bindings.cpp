#include "Colorspace_bindings.hpp"

namespace py = pybind11;

void bind_Colorspace(py::module_ &m) {
    py::class_<Colorspace, FilterBase, std::shared_ptr<Colorspace>>(m, "Colorspace")
        .def(py::init<int, int, int, int, int, int, bool, int, int, int, int, int, int, int>(),
             py::arg("all") = 0,
             py::arg("space") = 2,
             py::arg("range") = 0,
             py::arg("primaries") = 2,
             py::arg("trc") = 2,
             py::arg("format") = -1,
             py::arg("fast") = false,
             py::arg("dither") = 0,
             py::arg("wpadapt") = 0,
             py::arg("iall") = 0,
             py::arg("ispace") = 2,
             py::arg("irange") = 0,
             py::arg("iprimaries") = 2,
             py::arg("itrc") = 2)
        .def("setAll", &Colorspace::setAll)
        .def("getAll", &Colorspace::getAll)
        .def("setSpace", &Colorspace::setSpace)
        .def("getSpace", &Colorspace::getSpace)
        .def("setRange", &Colorspace::setRange)
        .def("getRange", &Colorspace::getRange)
        .def("setPrimaries", &Colorspace::setPrimaries)
        .def("getPrimaries", &Colorspace::getPrimaries)
        .def("setTrc", &Colorspace::setTrc)
        .def("getTrc", &Colorspace::getTrc)
        .def("setFormat", &Colorspace::setFormat)
        .def("getFormat", &Colorspace::getFormat)
        .def("setFast", &Colorspace::setFast)
        .def("getFast", &Colorspace::getFast)
        .def("setDither", &Colorspace::setDither)
        .def("getDither", &Colorspace::getDither)
        .def("setWpadapt", &Colorspace::setWpadapt)
        .def("getWpadapt", &Colorspace::getWpadapt)
        .def("setIall", &Colorspace::setIall)
        .def("getIall", &Colorspace::getIall)
        .def("setIspace", &Colorspace::setIspace)
        .def("getIspace", &Colorspace::getIspace)
        .def("setIrange", &Colorspace::setIrange)
        .def("getIrange", &Colorspace::getIrange)
        .def("setIprimaries", &Colorspace::setIprimaries)
        .def("getIprimaries", &Colorspace::getIprimaries)
        .def("setItrc", &Colorspace::setItrc)
        .def("getItrc", &Colorspace::getItrc)
        ;
}