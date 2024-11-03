#include "Noise_bindings.hpp"

namespace py = pybind11;

void bind_Noise(py::module_ &m) {
    py::class_<Noise, FilterBase, std::shared_ptr<Noise>>(m, "Noise")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int, int, int>(),
             py::arg("all_seed") = -1,
             py::arg("all_strength") = 0,
             py::arg("all_flags") = 0,
             py::arg("c0_flags") = 0,
             py::arg("c1_seed") = -1,
             py::arg("c1_strength") = 0,
             py::arg("c1_flags") = 0,
             py::arg("c2_seed") = -1,
             py::arg("c2_strength") = 0,
             py::arg("c2_flags") = 0,
             py::arg("c3_seed") = -1,
             py::arg("c3_strength") = 0,
             py::arg("c3_flags") = 0)
        .def("setAll_seed", &Noise::setAll_seed)
        .def("getAll_seed", &Noise::getAll_seed)
        .def("setAll_strength", &Noise::setAll_strength)
        .def("getAll_strength", &Noise::getAll_strength)
        .def("setAll_flags", &Noise::setAll_flags)
        .def("getAll_flags", &Noise::getAll_flags)
        .def("setC0_flags", &Noise::setC0_flags)
        .def("getC0_flags", &Noise::getC0_flags)
        .def("setC1_seed", &Noise::setC1_seed)
        .def("getC1_seed", &Noise::getC1_seed)
        .def("setC1_strength", &Noise::setC1_strength)
        .def("getC1_strength", &Noise::getC1_strength)
        .def("setC1_flags", &Noise::setC1_flags)
        .def("getC1_flags", &Noise::getC1_flags)
        .def("setC2_seed", &Noise::setC2_seed)
        .def("getC2_seed", &Noise::getC2_seed)
        .def("setC2_strength", &Noise::setC2_strength)
        .def("getC2_strength", &Noise::getC2_strength)
        .def("setC2_flags", &Noise::setC2_flags)
        .def("getC2_flags", &Noise::getC2_flags)
        .def("setC3_seed", &Noise::setC3_seed)
        .def("getC3_seed", &Noise::getC3_seed)
        .def("setC3_strength", &Noise::setC3_strength)
        .def("getC3_strength", &Noise::getC3_strength)
        .def("setC3_flags", &Noise::setC3_flags)
        .def("getC3_flags", &Noise::getC3_flags)
        ;
}