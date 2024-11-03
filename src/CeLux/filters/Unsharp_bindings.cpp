#include "Unsharp_bindings.hpp"

namespace py = pybind11;

void bind_Unsharp(py::module_ &m) {
    py::class_<Unsharp, FilterBase, std::shared_ptr<Unsharp>>(m, "Unsharp")
        .def(py::init<int, int, float, int, int, float, int, int, float>(),
             py::arg("luma_msize_x") = 5,
             py::arg("luma_msize_y") = 5,
             py::arg("luma_amount") = 1.00,
             py::arg("chroma_msize_x") = 5,
             py::arg("chroma_msize_y") = 5,
             py::arg("chroma_amount") = 0.00,
             py::arg("alpha_msize_x") = 5,
             py::arg("alpha_msize_y") = 5,
             py::arg("alpha_amount") = 0.00)
        .def("setLuma_msize_x", &Unsharp::setLuma_msize_x)
        .def("getLuma_msize_x", &Unsharp::getLuma_msize_x)
        .def("setLuma_msize_y", &Unsharp::setLuma_msize_y)
        .def("getLuma_msize_y", &Unsharp::getLuma_msize_y)
        .def("setLuma_amount", &Unsharp::setLuma_amount)
        .def("getLuma_amount", &Unsharp::getLuma_amount)
        .def("setChroma_msize_x", &Unsharp::setChroma_msize_x)
        .def("getChroma_msize_x", &Unsharp::getChroma_msize_x)
        .def("setChroma_msize_y", &Unsharp::setChroma_msize_y)
        .def("getChroma_msize_y", &Unsharp::getChroma_msize_y)
        .def("setChroma_amount", &Unsharp::setChroma_amount)
        .def("getChroma_amount", &Unsharp::getChroma_amount)
        .def("setAlpha_msize_x", &Unsharp::setAlpha_msize_x)
        .def("getAlpha_msize_x", &Unsharp::getAlpha_msize_x)
        .def("setAlpha_msize_y", &Unsharp::setAlpha_msize_y)
        .def("getAlpha_msize_y", &Unsharp::getAlpha_msize_y)
        .def("setAlpha_amount", &Unsharp::setAlpha_amount)
        .def("getAlpha_amount", &Unsharp::getAlpha_amount)
        ;
}