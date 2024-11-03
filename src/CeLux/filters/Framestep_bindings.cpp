#include "Framestep_bindings.hpp"

namespace py = pybind11;

void bind_Framestep(py::module_ &m) {
    py::class_<Framestep, FilterBase, std::shared_ptr<Framestep>>(m, "Framestep")
        .def(py::init<int>(),
             py::arg("step") = 1)
        .def("setStep", &Framestep::setStep)
        .def("getStep", &Framestep::getStep)
        ;
}