#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ProbDataAssociation.h>


namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(papy, m) {
    m.doc() = "probabilistic association python"; // optional module docstring
    m.def("compute_permanent", &permanentFastest, "A function which computes the permanent of a matrix",
          "mat"_a);
    m.def("compute_weights", &compute_weights, "A function which computes the association weights",
          "mat"_a);
}
