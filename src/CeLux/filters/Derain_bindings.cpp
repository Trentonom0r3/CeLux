#include "Derain_bindings.hpp"

namespace py = pybind11;

void bind_Derain(py::module_ &m) {
    py::class_<Derain, FilterBase, std::shared_ptr<Derain>>(m, "Derain")
        .def(py::init<int, int, std::string, std::string, std::string>(),
             py::arg("filter_type") = 0,
             py::arg("dnn_backend") = 1,
             py::arg("model") = "",
             py::arg("input") = "x",
             py::arg("output") = "y")
        .def("setFilter_type", &Derain::setFilter_type)
        .def("getFilter_type", &Derain::getFilter_type)
        .def("setDnn_backend", &Derain::setDnn_backend)
        .def("getDnn_backend", &Derain::getDnn_backend)
        .def("setModel", &Derain::setModel)
        .def("getModel", &Derain::getModel)
        .def("setInput", &Derain::setInput)
        .def("getInput", &Derain::getInput)
        .def("setOutput", &Derain::setOutput)
        .def("getOutput", &Derain::getOutput)
        ;
}