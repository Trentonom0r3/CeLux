#include "Pad_bindings.hpp"

namespace py = pybind11;

void bind_Pad(py::module_ &m) {
    py::class_<Pad, FilterBase, std::shared_ptr<Pad>>(m, "Pad")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, int, std::pair<int, int>>(),
             py::arg("width") = "iw",
             py::arg("height") = "ih",
             py::arg("xOffsetForTheInputImagePosition") = "0",
             py::arg("yOffsetForTheInputImagePosition") = "0",
             py::arg("color") = "black",
             py::arg("eval") = 0,
             py::arg("aspect") = std::make_pair<int, int>(0, 1))
        .def("setWidth", &Pad::setWidth)
        .def("getWidth", &Pad::getWidth)
        .def("setHeight", &Pad::setHeight)
        .def("getHeight", &Pad::getHeight)
        .def("setXOffsetForTheInputImagePosition", &Pad::setXOffsetForTheInputImagePosition)
        .def("getXOffsetForTheInputImagePosition", &Pad::getXOffsetForTheInputImagePosition)
        .def("setYOffsetForTheInputImagePosition", &Pad::setYOffsetForTheInputImagePosition)
        .def("getYOffsetForTheInputImagePosition", &Pad::getYOffsetForTheInputImagePosition)
        .def("setColor", &Pad::setColor)
        .def("getColor", &Pad::getColor)
        .def("setEval", &Pad::setEval)
        .def("getEval", &Pad::getEval)
        .def("setAspect", &Pad::setAspect)
        .def("getAspect", &Pad::getAspect)
        ;
}