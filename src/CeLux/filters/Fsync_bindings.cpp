#include "Fsync_bindings.hpp"

namespace py = pybind11;

void bind_Fsync(py::module_ &m) {
    py::class_<Fsync, FilterBase, std::shared_ptr<Fsync>>(m, "Fsync")
        .def(py::init<std::string>(),
             py::arg("file") = "")
        .def("setFile", &Fsync::setFile)
        .def("getFile", &Fsync::getFile)
        ;
}