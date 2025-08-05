TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(shell pwd)/distribution.v
TOPLEVEL = distribution
MODULE = test_distribution

# Add VCS flags to enable VCD dumping
COMPILE_ARGS += -debug_all
SIM_ARGS += -debug_all -vcs+dumpvars+simulation.vcd

# Include Cocotb Makefile
include $(shell cocotb-config --makefiles)/Makefile.sim
