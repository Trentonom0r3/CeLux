#include "Ciescope_bindings.hpp"

namespace py = pybind11;

void bind_Ciescope(py::module_ &m) {
    py::class_<Ciescope, FilterBase, std::shared_ptr<Ciescope>>(m, "Ciescope")
        .def(py::init<int, int, int, int, float, float, bool, bool, double, bool>(),
             py::arg("system") = 7,
             py::arg("cie") = 0,
             py::arg("gamuts") = 0,
             py::arg("size") = 512,
             py::arg("intensity") = 0.00,
             py::arg("contrast") = 0.75,
             py::arg("corrgamma") = true,
             py::arg("showwhite") = false,
             py::arg("gamma") = 2.60,
             py::arg("fill") = true)
        .def("setSystem", &Ciescope::setSystem)
        .def("getSystem", &Ciescope::getSystem)
        .def("setCie", &Ciescope::setCie)
        .def("getCie", &Ciescope::getCie)
        .def("setGamuts", &Ciescope::setGamuts)
        .def("getGamuts", &Ciescope::getGamuts)
        .def("setSize", &Ciescope::setSize)
        .def("getSize", &Ciescope::getSize)
        .def("setIntensity", &Ciescope::setIntensity)
        .def("getIntensity", &Ciescope::getIntensity)
        .def("setContrast", &Ciescope::setContrast)
        .def("getContrast", &Ciescope::getContrast)
        .def("setCorrgamma", &Ciescope::setCorrgamma)
        .def("getCorrgamma", &Ciescope::getCorrgamma)
        .def("setShowwhite", &Ciescope::setShowwhite)
        .def("getShowwhite", &Ciescope::getShowwhite)
        .def("setGamma", &Ciescope::setGamma)
        .def("getGamma", &Ciescope::getGamma)
        .def("setFill", &Ciescope::setFill)
        .def("getFill", &Ciescope::getFill)
        ;
}