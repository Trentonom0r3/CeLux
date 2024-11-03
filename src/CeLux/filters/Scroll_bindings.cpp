#include "Scroll_bindings.hpp"

namespace py = pybind11;

void bind_Scroll(py::module_ &m) {
    py::class_<Scroll, FilterBase, std::shared_ptr<Scroll>>(m, "Scroll")
        .def(py::init<float, float, float, float>(),
             py::arg("horizontal") = 0.00,
             py::arg("vertical") = 0.00,
             py::arg("hpos") = 0.00,
             py::arg("vpos") = 0.00)
        .def("setHorizontal", &Scroll::setHorizontal)
        .def("getHorizontal", &Scroll::getHorizontal)
        .def("setVertical", &Scroll::setVertical)
        .def("getVertical", &Scroll::getVertical)
        .def("setHpos", &Scroll::setHpos)
        .def("getHpos", &Scroll::getHpos)
        .def("setVpos", &Scroll::setVpos)
        .def("getVpos", &Scroll::getVpos)
        ;
}