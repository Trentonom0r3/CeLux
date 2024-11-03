#include "Xmedian_bindings.hpp"

namespace py = pybind11;

void bind_Xmedian(py::module_ &m) {
    py::class_<Xmedian, FilterBase, std::shared_ptr<Xmedian>>(m, "Xmedian")
        .def(py::init<int, int, float>(),
             py::arg("inputs") = 3,
             py::arg("planes") = 15,
             py::arg("percentile") = 0.50)
        .def("setInputs", &Xmedian::setInputs)
        .def("getInputs", &Xmedian::getInputs)
        .def("setPlanes", &Xmedian::setPlanes)
        .def("getPlanes", &Xmedian::getPlanes)
        .def("setPercentile", &Xmedian::setPercentile)
        .def("getPercentile", &Xmedian::getPercentile)
        ;
}