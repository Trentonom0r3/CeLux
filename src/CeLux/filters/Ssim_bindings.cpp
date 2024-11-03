#include "Ssim_bindings.hpp"

namespace py = pybind11;

void bind_Ssim(py::module_ &m) {
    py::class_<Ssim, FilterBase, std::shared_ptr<Ssim>>(m, "Ssim")
        .def(py::init<std::string>(),
             py::arg("stats_file") = "")
        .def("setStats_file", &Ssim::setStats_file)
        .def("getStats_file", &Ssim::getStats_file)
        ;
}