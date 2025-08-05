`timescale 1ns / 1ps

// Currently combinational, but two mux stages can be also pipelined
module distribution #(
    parameter DATA_WIDTH = 18, // Assuming 9-bit input activation and/or weights, clubbed data
    parameter NUM_INPUT_DATA = 4,
    parameter NUM_SUB_MACROS = 4
) 
(
    clk,  // clock
    rst_n,  // Negative edge reset

    i_valid,
    i_data_bus,

    // control signals
    i_en,  // distribute switch enable
    i_sparse_select, // Mux control logic
    o_valid,  // output valid
    o_data_bus  // output data

);

  // timing signals
  input clk;
  input rst_n;

  // interface
  input i_valid;
  input [NUM_SUB_MACROS*NUM_INPUT_DATA*DATA_WIDTH-1:0] i_data_bus;

  output  o_valid;
  output [NUM_SUB_MACROS*DATA_WIDTH-1:0] o_data_bus;  

  input i_en;
  input [2*NUM_SUB_MACROS*2-1 : 0] i_sparse_select; // 2 rows of MUX and each mux requires 2-bit select

  wire [NUM_SUB_MACROS * DATA_WIDTH-1:0] mux1_out;
  wire [NUM_SUB_MACROS * DATA_WIDTH-1:0] mux2_out;

  wire [1:0] select1 [NUM_SUB_MACROS-1:0];
  wire [1:0] select2 [NUM_SUB_MACROS-1:0];

  // First layer of 4:1 muxes
  genvar i, j;
  generate
    for (i = 0; i < NUM_SUB_MACROS; i = i + 1) begin : gen_mux1
        assign select1[i] = i_sparse_select[2*i +: 2];
        assign mux1_out[i*DATA_WIDTH +: DATA_WIDTH] = i_data_bus[(i*NUM_INPUT_DATA*DATA_WIDTH + select1[i]*DATA_WIDTH )+: DATA_WIDTH];
    end

    // Second layer of 4:1 mux
    for (j = 0; j < NUM_SUB_MACROS; j = j + 1) begin : gen_mux2_inputs
        assign select2[j] = i_sparse_select[(NUM_SUB_MACROS*2 + 2*j) +: 2];
        assign mux2_out[j*DATA_WIDTH +: DATA_WIDTH] = mux1_out[select2[j]*DATA_WIDTH +: DATA_WIDTH];

    end
  endgenerate


  // Final output assignment
  assign o_data_bus = mux2_out;
  assign o_valid = i_valid;


endmodule
