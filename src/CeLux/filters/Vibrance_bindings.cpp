#include "Vibrance_bindings.hpp"

namespace py = pybind11;

void bind_Vibrance(py::module_ &m) {
    py::class_<Vibrance, FilterBase, std::shared_ptr<Vibrance>>(m, "Vibrance")
        .def(py::init<float, float, float, float, float, float, float, bool>(),
             py::arg("intensity") = 0.00,
             py::arg("rbal") = 1.00,
             py::arg("gbal") = 1.00,
             py::arg("bbal") = 1.00,
             py::arg("rlum") = 0.07,
             py::arg("glum") = 0.72,
             py::arg("blum") = 0.21,
             py::arg("alternate") = false)
        .def("setIntensity", &Vibrance::setIntensity)
        .def("getIntensity", &Vibrance::getIntensity)
        .def("setRbal", &Vibrance::setRbal)
        .def("getRbal", &Vibrance::getRbal)
        .def("setGbal", &Vibrance::setGbal)
        .def("getGbal", &Vibrance::getGbal)
        .def("setBbal", &Vibrance::setBbal)
        .def("getBbal", &Vibrance::getBbal)
        .def("setRlum", &Vibrance::setRlum)
        .def("getRlum", &Vibrance::getRlum)
        .def("setGlum", &Vibrance::setGlum)
        .def("getGlum", &Vibrance::getGlum)
        .def("setBlum", &Vibrance::setBlum)
        .def("getBlum", &Vibrance::getBlum)
        .def("setAlternate", &Vibrance::setAlternate)
        .def("getAlternate", &Vibrance::getAlternate)
        ;
}