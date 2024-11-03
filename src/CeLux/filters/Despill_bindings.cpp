#include "Despill_bindings.hpp"

namespace py = pybind11;

void bind_Despill(py::module_ &m) {
    py::class_<Despill, FilterBase, std::shared_ptr<Despill>>(m, "Despill")
        .def(py::init<int, float, float, float, float, float, float, bool>(),
             py::arg("type") = 0,
             py::arg("mix") = 0.50,
             py::arg("expand") = 0.00,
             py::arg("red") = 0.00,
             py::arg("green") = -1.00,
             py::arg("blue") = 0.00,
             py::arg("brightness") = 0.00,
             py::arg("alpha") = false)
        .def("setType", &Despill::setType)
        .def("getType", &Despill::getType)
        .def("setMix", &Despill::setMix)
        .def("getMix", &Despill::getMix)
        .def("setExpand", &Despill::setExpand)
        .def("getExpand", &Despill::getExpand)
        .def("setRed", &Despill::setRed)
        .def("getRed", &Despill::getRed)
        .def("setGreen", &Despill::setGreen)
        .def("getGreen", &Despill::getGreen)
        .def("setBlue", &Despill::setBlue)
        .def("getBlue", &Despill::getBlue)
        .def("setBrightness", &Despill::setBrightness)
        .def("getBrightness", &Despill::getBrightness)
        .def("setAlpha", &Despill::setAlpha)
        .def("getAlpha", &Despill::getAlpha)
        ;
}