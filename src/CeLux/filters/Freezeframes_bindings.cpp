#include "Freezeframes_bindings.hpp"

namespace py = pybind11;

void bind_Freezeframes(py::module_ &m) {
    py::class_<Freezeframes, FilterBase, std::shared_ptr<Freezeframes>>(m, "Freezeframes")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("first") = 0ULL,
             py::arg("last") = 0ULL,
             py::arg("replace") = 0ULL)
        .def("setFirst", &Freezeframes::setFirst)
        .def("getFirst", &Freezeframes::getFirst)
        .def("setLast", &Freezeframes::setLast)
        .def("getLast", &Freezeframes::getLast)
        .def("setReplace", &Freezeframes::setReplace)
        .def("getReplace", &Freezeframes::getReplace)
        ;
}