#include "Shuffleframes_bindings.hpp"

namespace py = pybind11;

void bind_Shuffleframes(py::module_ &m) {
    py::class_<Shuffleframes, FilterBase, std::shared_ptr<Shuffleframes>>(m, "Shuffleframes")
        .def(py::init<std::string>(),
             py::arg("mapping") = "0")
        .def("setMapping", &Shuffleframes::setMapping)
        .def("getMapping", &Shuffleframes::getMapping)
        ;
}