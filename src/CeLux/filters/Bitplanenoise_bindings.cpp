#include "Bitplanenoise_bindings.hpp"

namespace py = pybind11;

void bind_Bitplanenoise(py::module_ &m) {
    py::class_<Bitplanenoise, FilterBase, std::shared_ptr<Bitplanenoise>>(m, "Bitplanenoise")
        .def(py::init<int, bool>(),
             py::arg("bitplane") = 1,
             py::arg("filter") = false)
        .def("setBitplane", &Bitplanenoise::setBitplane)
        .def("getBitplane", &Bitplanenoise::getBitplane)
        .def("setFilter", &Bitplanenoise::setFilter)
        .def("getFilter", &Bitplanenoise::getFilter)
        ;
}