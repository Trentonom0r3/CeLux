#include "Elbg_bindings.hpp"

namespace py = pybind11;

void bind_Elbg(py::module_ &m) {
    py::class_<Elbg, FilterBase, std::shared_ptr<Elbg>>(m, "Elbg")
        .def(py::init<int, int, int64_t, bool, bool>(),
             py::arg("codebook_length") = 256,
             py::arg("nb_steps") = 1,
             py::arg("seed") = 0,
             py::arg("pal8") = false,
             py::arg("use_alpha") = false)
        .def("setCodebook_length", &Elbg::setCodebook_length)
        .def("getCodebook_length", &Elbg::getCodebook_length)
        .def("setNb_steps", &Elbg::setNb_steps)
        .def("getNb_steps", &Elbg::getNb_steps)
        .def("setSeed", &Elbg::setSeed)
        .def("getSeed", &Elbg::getSeed)
        .def("setPal8", &Elbg::setPal8)
        .def("getPal8", &Elbg::getPal8)
        .def("setUse_alpha", &Elbg::setUse_alpha)
        .def("getUse_alpha", &Elbg::getUse_alpha)
        ;
}