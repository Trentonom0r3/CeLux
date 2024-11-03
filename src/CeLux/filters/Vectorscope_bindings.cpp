#include "Vectorscope_bindings.hpp"

namespace py = pybind11;

void bind_Vectorscope(py::module_ &m) {
    py::class_<Vectorscope, FilterBase, std::shared_ptr<Vectorscope>>(m, "Vectorscope")
        .def(py::init<int, int, int, float, int, int, float, int, float, float, float, int, float, float>(),
             py::arg("mode") = 0,
             py::arg("colorComponentOnXAxis") = 1,
             py::arg("colorComponentOnYAxis") = 2,
             py::arg("intensity") = 0.00,
             py::arg("envelope") = 0,
             py::arg("graticule") = 0,
             py::arg("opacity") = 0.75,
             py::arg("flags") = 4,
             py::arg("bgopacity") = 0.30,
             py::arg("lthreshold") = 0.00,
             py::arg("hthreshold") = 1.00,
             py::arg("colorspace") = 0,
             py::arg("tint0") = 0.00,
             py::arg("tint1") = 0.00)
        .def("setMode", &Vectorscope::setMode)
        .def("getMode", &Vectorscope::getMode)
        .def("setColorComponentOnXAxis", &Vectorscope::setColorComponentOnXAxis)
        .def("getColorComponentOnXAxis", &Vectorscope::getColorComponentOnXAxis)
        .def("setColorComponentOnYAxis", &Vectorscope::setColorComponentOnYAxis)
        .def("getColorComponentOnYAxis", &Vectorscope::getColorComponentOnYAxis)
        .def("setIntensity", &Vectorscope::setIntensity)
        .def("getIntensity", &Vectorscope::getIntensity)
        .def("setEnvelope", &Vectorscope::setEnvelope)
        .def("getEnvelope", &Vectorscope::getEnvelope)
        .def("setGraticule", &Vectorscope::setGraticule)
        .def("getGraticule", &Vectorscope::getGraticule)
        .def("setOpacity", &Vectorscope::setOpacity)
        .def("getOpacity", &Vectorscope::getOpacity)
        .def("setFlags", &Vectorscope::setFlags)
        .def("getFlags", &Vectorscope::getFlags)
        .def("setBgopacity", &Vectorscope::setBgopacity)
        .def("getBgopacity", &Vectorscope::getBgopacity)
        .def("setLthreshold", &Vectorscope::setLthreshold)
        .def("getLthreshold", &Vectorscope::getLthreshold)
        .def("setHthreshold", &Vectorscope::setHthreshold)
        .def("getHthreshold", &Vectorscope::getHthreshold)
        .def("setColorspace", &Vectorscope::setColorspace)
        .def("getColorspace", &Vectorscope::getColorspace)
        .def("setTint0", &Vectorscope::setTint0)
        .def("getTint0", &Vectorscope::getTint0)
        .def("setTint1", &Vectorscope::setTint1)
        .def("getTint1", &Vectorscope::getTint1)
        ;
}