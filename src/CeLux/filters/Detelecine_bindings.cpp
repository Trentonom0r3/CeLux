#include "Detelecine_bindings.hpp"

namespace py = pybind11;

void bind_Detelecine(py::module_ &m) {
    py::class_<Detelecine, FilterBase, std::shared_ptr<Detelecine>>(m, "Detelecine")
        .def(py::init<int, std::string, int>(),
             py::arg("first_field") = 0,
             py::arg("pattern") = "23",
             py::arg("start_frame") = 0)
        .def("setFirst_field", &Detelecine::setFirst_field)
        .def("getFirst_field", &Detelecine::getFirst_field)
        .def("setPattern", &Detelecine::setPattern)
        .def("getPattern", &Detelecine::getPattern)
        .def("setStart_frame", &Detelecine::setStart_frame)
        .def("getStart_frame", &Detelecine::getStart_frame)
        ;
}