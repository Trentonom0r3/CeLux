#include "Oscilloscope_bindings.hpp"

namespace py = pybind11;

void bind_Oscilloscope(py::module_ &m) {
    py::class_<Oscilloscope, FilterBase, std::shared_ptr<Oscilloscope>>(m, "Oscilloscope")
        .def(py::init<float, float, float, float, float, float, float, float, float, int, bool, bool, bool>(),
             py::arg("scopeXPosition") = 0.50,
             py::arg("scopeYPosition") = 0.50,
             py::arg("scopeSize") = 0.80,
             py::arg("scopeTilt") = 0.50,
             py::arg("traceOpacity") = 0.80,
             py::arg("tx") = 0.50,
             py::arg("ty") = 0.90,
             py::arg("tw") = 0.80,
             py::arg("th") = 0.30,
             py::arg("componentsToTrace") = 7,
             py::arg("drawTraceGrid") = true,
             py::arg("st") = true,
             py::arg("sc") = true)
        .def("setScopeXPosition", &Oscilloscope::setScopeXPosition)
        .def("getScopeXPosition", &Oscilloscope::getScopeXPosition)
        .def("setScopeYPosition", &Oscilloscope::setScopeYPosition)
        .def("getScopeYPosition", &Oscilloscope::getScopeYPosition)
        .def("setScopeSize", &Oscilloscope::setScopeSize)
        .def("getScopeSize", &Oscilloscope::getScopeSize)
        .def("setScopeTilt", &Oscilloscope::setScopeTilt)
        .def("getScopeTilt", &Oscilloscope::getScopeTilt)
        .def("setTraceOpacity", &Oscilloscope::setTraceOpacity)
        .def("getTraceOpacity", &Oscilloscope::getTraceOpacity)
        .def("setTx", &Oscilloscope::setTx)
        .def("getTx", &Oscilloscope::getTx)
        .def("setTy", &Oscilloscope::setTy)
        .def("getTy", &Oscilloscope::getTy)
        .def("setTw", &Oscilloscope::setTw)
        .def("getTw", &Oscilloscope::getTw)
        .def("setTh", &Oscilloscope::setTh)
        .def("getTh", &Oscilloscope::getTh)
        .def("setComponentsToTrace", &Oscilloscope::setComponentsToTrace)
        .def("getComponentsToTrace", &Oscilloscope::getComponentsToTrace)
        .def("setDrawTraceGrid", &Oscilloscope::setDrawTraceGrid)
        .def("getDrawTraceGrid", &Oscilloscope::getDrawTraceGrid)
        .def("setSt", &Oscilloscope::setSt)
        .def("getSt", &Oscilloscope::getSt)
        .def("setSc", &Oscilloscope::setSc)
        .def("getSc", &Oscilloscope::getSc)
        ;
}