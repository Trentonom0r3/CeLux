#include "Vignette_bindings.hpp"

namespace py = pybind11;

void bind_Vignette(py::module_ &m) {
    py::class_<Vignette, FilterBase, std::shared_ptr<Vignette>>(m, "Vignette")
        .def(py::init<std::string, std::string, std::string, int, int, bool, std::pair<int, int>>(),
             py::arg("angle") = "PI/5",
             py::arg("x0") = "w/2",
             py::arg("y0") = "h/2",
             py::arg("mode") = 0,
             py::arg("eval") = 0,
             py::arg("dither") = true,
             py::arg("aspect") = std::make_pair<int, int>(0, 1))
        .def("setAngle", &Vignette::setAngle)
        .def("getAngle", &Vignette::getAngle)
        .def("setX0", &Vignette::setX0)
        .def("getX0", &Vignette::getX0)
        .def("setY0", &Vignette::setY0)
        .def("getY0", &Vignette::getY0)
        .def("setMode", &Vignette::setMode)
        .def("getMode", &Vignette::getMode)
        .def("setEval", &Vignette::setEval)
        .def("getEval", &Vignette::getEval)
        .def("setDither", &Vignette::setDither)
        .def("getDither", &Vignette::getDither)
        .def("setAspect", &Vignette::setAspect)
        .def("getAspect", &Vignette::getAspect)
        ;
}