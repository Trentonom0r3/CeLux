#include "Interleave_bindings.hpp"

namespace py = pybind11;

void bind_Interleave(py::module_ &m) {
    py::class_<Interleave, FilterBase, std::shared_ptr<Interleave>>(m, "Interleave")
        .def(py::init<int, int>(),
             py::arg("nb_inputs") = 2,
             py::arg("duration") = 0)
        .def("setNb_inputs", &Interleave::setNb_inputs)
        .def("getNb_inputs", &Interleave::getNb_inputs)
        .def("setDuration", &Interleave::setDuration)
        .def("getDuration", &Interleave::getDuration)
        ;
}