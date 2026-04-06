#include "pseudo_channel.hpp"
#include "ramulator_hbm.hpp"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(pim_bindings, m) {
  m.doc() = "PIM-graphs Ramulator HBM driver (optional build)";

  py::class_<pim::SimulationResult>(m, "SimulationResult")
      .def_readonly("memory_cycles", &pim::SimulationResult::memory_cycles)
      .def_readonly("bytes_moved", &pim::SimulationResult::bytes_moved)
      .def_readonly("bursts_completed", &pim::SimulationResult::bursts_completed)
      .def_readonly("tck_ns", &pim::SimulationResult::tck_ns)
      .def_readonly("tsv_bytes_admitted", &pim::SimulationResult::tsv_bytes_admitted)
      .def_readonly("tsv_peak_gbps", &pim::SimulationResult::tsv_peak_gbps)
      .def("effective_gbps", &pim::SimulationResult::effective_gbps);

  m.def(
      "run_sequential_16pc",
      [](const std::string& yaml_path, std::uint32_t bursts_per_pc, std::uint64_t max_cycles,
         double tsv_peak_gbps) {
        pim::RamulatorHbmSimulator sim(yaml_path);
        sim.set_tsv_peak_gbps(tsv_peak_gbps);
        sim.mux().load_sequential_workload(bursts_per_pc, 64);
        pim::SimulationResult r = sim.run(max_cycles);
        sim.finalize();
        return r;
      },
      py::arg("config_path"), py::arg("bursts_per_pc") = 8,
      py::arg("max_memory_cycles") = 500000000ull, py::arg("tsv_peak_gbps") = 512.0);
}
