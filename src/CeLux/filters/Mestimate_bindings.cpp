#include "Mestimate_bindings.hpp"

namespace py = pybind11;

void bind_Mestimate(py::module_ &m) {
    py::class_<Mestimate, FilterBase, std::shared_ptr<Mestimate>>(m, "Mestimate")
        .def(py::init<int, int, int>(),
             py::arg("method") = 1,
             py::arg("mb_size") = 16,
             py::arg("search_param") = 7)
        .def("setMethod", &Mestimate::setMethod)
        .def("getMethod", &Mestimate::getMethod)
        .def("setMb_size", &Mestimate::setMb_size)
        .def("getMb_size", &Mestimate::getMb_size)
        .def("setSearch_param", &Mestimate::setSearch_param)
        .def("getSearch_param", &Mestimate::getSearch_param)
        ;
}