#include "Pixscope_bindings.hpp"

namespace py = pybind11;

void bind_Pixscope(py::module_ &m) {
    py::class_<Pixscope, FilterBase, std::shared_ptr<Pixscope>>(m, "Pixscope")
        .def(py::init<float, float, int, int, float, float, float>(),
             py::arg("scopeXOffset") = 0.50,
             py::arg("scopeYOffset") = 0.50,
             py::arg("scopeWidth") = 7,
             py::arg("scopeHeight") = 7,
             py::arg("windowOpacity") = 0.50,
             py::arg("wx") = -1.00,
             py::arg("wy") = -1.00)
        .def("setScopeXOffset", &Pixscope::setScopeXOffset)
        .def("getScopeXOffset", &Pixscope::getScopeXOffset)
        .def("setScopeYOffset", &Pixscope::setScopeYOffset)
        .def("getScopeYOffset", &Pixscope::getScopeYOffset)
        .def("setScopeWidth", &Pixscope::setScopeWidth)
        .def("getScopeWidth", &Pixscope::getScopeWidth)
        .def("setScopeHeight", &Pixscope::setScopeHeight)
        .def("getScopeHeight", &Pixscope::getScopeHeight)
        .def("setWindowOpacity", &Pixscope::setWindowOpacity)
        .def("getWindowOpacity", &Pixscope::getWindowOpacity)
        .def("setWx", &Pixscope::setWx)
        .def("getWx", &Pixscope::getWx)
        .def("setWy", &Pixscope::setWy)
        .def("getWy", &Pixscope::getWy)
        ;
}