#include "Pixdesctest_bindings.hpp"

namespace py = pybind11;

void bind_Pixdesctest(py::module_ &m) {
    py::class_<Pixdesctest, FilterBase, std::shared_ptr<Pixdesctest>>(m, "Pixdesctest")
        .def(py::init<>())
        ;
}