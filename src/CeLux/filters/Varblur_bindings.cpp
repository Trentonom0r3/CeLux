#include "Varblur_bindings.hpp"

namespace py = pybind11;

void bind_Varblur(py::module_ &m) {
    py::class_<Varblur, FilterBase, std::shared_ptr<Varblur>>(m, "Varblur")
        .def(py::init<int, int, int>(),
             py::arg("min_r") = 0,
             py::arg("max_r") = 8,
             py::arg("planes") = 15)
        .def("setMin_r", &Varblur::setMin_r)
        .def("getMin_r", &Varblur::getMin_r)
        .def("setMax_r", &Varblur::setMax_r)
        .def("getMax_r", &Varblur::getMax_r)
        .def("setPlanes", &Varblur::setPlanes)
        .def("getPlanes", &Varblur::getPlanes)
        ;
}