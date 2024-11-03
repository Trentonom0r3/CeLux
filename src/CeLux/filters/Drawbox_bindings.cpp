#include "Drawbox_bindings.hpp"

namespace py = pybind11;

void bind_Drawbox(py::module_ &m) {
    py::class_<Drawbox, FilterBase, std::shared_ptr<Drawbox>>(m, "Drawbox")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string, bool, std::string>(),
             py::arg("horizontalPositionOfTheLeftBoxEdge") = "0",
             py::arg("verticalPositionOfTheTopBoxEdge") = "0",
             py::arg("width") = "0",
             py::arg("height") = "0",
             py::arg("color") = "black",
             py::arg("thickness") = "3",
             py::arg("replace") = false,
             py::arg("box_source") = "")
        .def("setHorizontalPositionOfTheLeftBoxEdge", &Drawbox::setHorizontalPositionOfTheLeftBoxEdge)
        .def("getHorizontalPositionOfTheLeftBoxEdge", &Drawbox::getHorizontalPositionOfTheLeftBoxEdge)
        .def("setVerticalPositionOfTheTopBoxEdge", &Drawbox::setVerticalPositionOfTheTopBoxEdge)
        .def("getVerticalPositionOfTheTopBoxEdge", &Drawbox::getVerticalPositionOfTheTopBoxEdge)
        .def("setWidth", &Drawbox::setWidth)
        .def("getWidth", &Drawbox::getWidth)
        .def("setHeight", &Drawbox::setHeight)
        .def("getHeight", &Drawbox::getHeight)
        .def("setColor", &Drawbox::setColor)
        .def("getColor", &Drawbox::getColor)
        .def("setThickness", &Drawbox::setThickness)
        .def("getThickness", &Drawbox::getThickness)
        .def("setReplace", &Drawbox::setReplace)
        .def("getReplace", &Drawbox::getReplace)
        .def("setBox_source", &Drawbox::setBox_source)
        .def("getBox_source", &Drawbox::getBox_source)
        ;
}