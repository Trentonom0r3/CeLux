#include "Hwupload_cuda_bindings.hpp"

namespace py = pybind11;

void bind_Hwupload_cuda(py::module_ &m) {
    py::class_<Hwupload_cuda, FilterBase, std::shared_ptr<Hwupload_cuda>>(m, "Hwupload_cuda")
        .def(py::init<int>(),
             py::arg("device") = 0)
        .def("setDevice", &Hwupload_cuda::setDevice)
        .def("getDevice", &Hwupload_cuda::getDevice)
        ;
}