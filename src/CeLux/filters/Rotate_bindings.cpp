#include "Rotate_bindings.hpp"

namespace py = pybind11;

void bind_Rotate(py::module_ &m) {
    py::class_<Rotate, FilterBase, std::shared_ptr<Rotate>>(m, "Rotate")
        .def(py::init<std::string, std::string, std::string, std::string, bool>(),
             py::arg("angle") = "0",
             py::arg("out_w") = "iw",
             py::arg("out_h") = "ih",
             py::arg("fillcolor") = "black",
             py::arg("bilinear") = true)
        .def("setAngle", &Rotate::setAngle)
        .def("getAngle", &Rotate::getAngle)
        .def("setOut_w", &Rotate::setOut_w)
        .def("getOut_w", &Rotate::getOut_w)
        .def("setOut_h", &Rotate::setOut_h)
        .def("getOut_h", &Rotate::getOut_h)
        .def("setFillcolor", &Rotate::setFillcolor)
        .def("getFillcolor", &Rotate::getFillcolor)
        .def("setBilinear", &Rotate::setBilinear)
        .def("getBilinear", &Rotate::getBilinear)
        ;
}