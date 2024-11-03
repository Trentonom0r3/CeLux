#include "Hwupload_bindings.hpp"

namespace py = pybind11;

void bind_Hwupload(py::module_ &m) {
    py::class_<Hwupload, FilterBase, std::shared_ptr<Hwupload>>(m, "Hwupload")
        .def(py::init<std::string>(),
             py::arg("derive_device") = "")
        .def("setDerive_device", &Hwupload::setDerive_device)
        .def("getDerive_device", &Hwupload::getDerive_device)
        ;
}