import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random

# Parameters
DATA_WIDTH = 16
NUM_INPUT_DATA = 16
NUM_MUX = 4
NUM_INPUT_BITS = NUM_MUX * NUM_INPUT_DATA * DATA_WIDTH

# Pre-defined arrays of 8-bit elements (64 elements per array)
array1 = [i for i in range(16)]
array2 = [i + 16 for i in range(16)]
array3 = [i + 32 for i in range(16)]
array4 = [i + 64 for i in range(16)]

# Function to extract a 4:1 mux output based on select signal
def mux_output(data, select):
    return data[select]

# Function to generate expected output based on the select signals
def calculate_expected_output(data_bus, sparse_select):
    expected_output = []
    layer1_output = []
    # Process first and second stage mux outputs
    mux1_0 = data_bus[sparse_select[0]]
    layer1_output.append(mux1_0)
    mux1_1 = data_bus[sparse_select[1] + 4]
    layer1_output.append(mux1_1)
    mux1_2 = data_bus[sparse_select[2] + 8]
    layer1_output.append(mux1_2)
    mux1_3 = data_bus[sparse_select[3] + 12]
    layer1_output.append(mux1_3)

    mux2_0 = layer1_output[sparse_select[4]]
    expected_output.append(mux2_0)
    mux2_1 = layer1_output[sparse_select[5]]
    expected_output.append(mux2_1)
    mux2_2 = layer1_output[sparse_select[6]]
    expected_output.append(mux2_2)
    mux2_3 = layer1_output[sparse_select[7]]
    expected_output.append(mux2_3)

    print("Expected: ", expected_output)
    return expected_output

# Testbench for the Verilog distribution module
@cocotb.test()
async def test_distribution(dut):
    # Generate the clock
    clock = Clock(dut.clk, 10, units="ns")  # Create a clock with 10ns period
    cocotb.start_soon(clock.start())  # Start the clock

    # Reset the DUT
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)  # Wait for a clock cycle
    dut.rst_n.value = 1  # Release reset
    await RisingEdge(dut.clk)  # Wait for another clock cycle


    # Combine the packaged data for the full input data bus
    combined_data_bus = array1 + array2 + array3

    # Run a test where we provide input values from the arrays and check the output
    for i in range(len(combined_data_bus) // NUM_INPUT_DATA):
        dut.i_valid.value = 1  # Set valid input to 1
        dut.i_en.value = 1  # Enable distribute switch

        # Assign the pre-packaged data to the input data bus (each iteration sends a slice of the full bus)
        start_idx = i * NUM_INPUT_DATA
        data_slice = combined_data_bus[start_idx:start_idx+NUM_INPUT_DATA]
        print(data_slice)
        dut.i_data_bus.value = int(''.join(format(x, f'0{DATA_WIDTH}b') for x in reversed(data_slice)), 2)
        # Generate random sparse select signal
        select_signal = [2, 3, 1, 2, 0, 0, 3, 1]
        print(select_signal)
        dut.i_sparse_select.value = int(''.join(format(x, '02b') for x in reversed(select_signal)), 2)

        await RisingEdge(dut.clk)  # Wait for the clock cycle

        # Extract output from DUT
        o_data_bus = int(dut.o_data_bus.value)
        # print(o_data_bus, "Output is here")
        # Calculate expected output based on the select signal
        expected_output = calculate_expected_output(
            data_slice,  # repeat the slice for all muxes
            select_signal
        )
        print("EXP", expected_output)

        # Compare the actual and expected output
        for j in range(NUM_MUX):
            actual_output = (o_data_bus >> (j * DATA_WIDTH)) & ((1 << DATA_WIDTH) - 1)
            print("Actual", actual_output)
            assert actual_output == expected_output[j], \
                f"Error: Mux {j} output is {actual_output}, but expected {expected_output[j]}"

        print(f"Output Data Bus: {o_data_bus} - Expected: {expected_output}")

    # After the test, disable the enable signal and valid
    dut.i_en.value = 0
    dut.i_valid.value = 0

    await RisingEdge(dut.clk)  # Ensure it completes with a clock edge
