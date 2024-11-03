#include "Drawgrid_bindings.hpp"

namespace py = pybind11;

void bind_Drawgrid(py::module_ &m) {
    py::class_<Drawgrid, FilterBase, std::shared_ptr<Drawgrid>>(m, "Drawgrid")
        .def(py::init<std::string, std::string, std::string, std::string, std::string, std::string, bool>(),
             py::arg("horizontalOffset") = "0",
             py::arg("verticalOffset") = "0",
             py::arg("width") = "0",
             py::arg("height") = "0",
             py::arg("color") = "black",
             py::arg("thickness") = "1",
             py::arg("replace") = false)
        .def("setHorizontalOffset", &Drawgrid::setHorizontalOffset)
        .def("getHorizontalOffset", &Drawgrid::getHorizontalOffset)
        .def("setVerticalOffset", &Drawgrid::setVerticalOffset)
        .def("getVerticalOffset", &Drawgrid::getVerticalOffset)
        .def("setWidth", &Drawgrid::setWidth)
        .def("getWidth", &Drawgrid::getWidth)
        .def("setHeight", &Drawgrid::setHeight)
        .def("getHeight", &Drawgrid::getHeight)
        .def("setColor", &Drawgrid::setColor)
        .def("getColor", &Drawgrid::getColor)
        .def("setThickness", &Drawgrid::setThickness)
        .def("getThickness", &Drawgrid::getThickness)
        .def("setReplace", &Drawgrid::setReplace)
        .def("getReplace", &Drawgrid::getReplace)
        ;
}