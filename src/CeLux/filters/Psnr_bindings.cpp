#include "Psnr_bindings.hpp"

namespace py = pybind11;

void bind_Psnr(py::module_ &m) {
    py::class_<Psnr, FilterBase, std::shared_ptr<Psnr>>(m, "Psnr")
        .def(py::init<std::string, int, bool>(),
             py::arg("stats_file") = "",
             py::arg("stats_version") = 1,
             py::arg("output_max") = false)
        .def("setStats_file", &Psnr::setStats_file)
        .def("getStats_file", &Psnr::getStats_file)
        .def("setStats_version", &Psnr::setStats_version)
        .def("getStats_version", &Psnr::getStats_version)
        .def("setOutput_max", &Psnr::setOutput_max)
        .def("getOutput_max", &Psnr::getOutput_max)
        ;
}