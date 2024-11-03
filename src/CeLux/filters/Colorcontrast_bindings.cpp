#include "Colorcontrast_bindings.hpp"

namespace py = pybind11;

void bind_Colorcontrast(py::module_ &m) {
    py::class_<Colorcontrast, FilterBase, std::shared_ptr<Colorcontrast>>(m, "Colorcontrast")
        .def(py::init<float, float, float, float, float, float, float>(),
             py::arg("rc") = 0.00,
             py::arg("gm") = 0.00,
             py::arg("by") = 0.00,
             py::arg("rcw") = 0.00,
             py::arg("gmw") = 0.00,
             py::arg("byw") = 0.00,
             py::arg("pl") = 0.00)
        .def("setRc", &Colorcontrast::setRc)
        .def("getRc", &Colorcontrast::getRc)
        .def("setGm", &Colorcontrast::setGm)
        .def("getGm", &Colorcontrast::getGm)
        .def("setBy", &Colorcontrast::setBy)
        .def("getBy", &Colorcontrast::getBy)
        .def("setRcw", &Colorcontrast::setRcw)
        .def("getRcw", &Colorcontrast::getRcw)
        .def("setGmw", &Colorcontrast::setGmw)
        .def("getGmw", &Colorcontrast::getGmw)
        .def("setByw", &Colorcontrast::setByw)
        .def("getByw", &Colorcontrast::getByw)
        .def("setPl", &Colorcontrast::setPl)
        .def("getPl", &Colorcontrast::getPl)
        ;
}