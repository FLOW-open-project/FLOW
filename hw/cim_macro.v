module cim_macro #(
    parameter DATA_WIDTH = 9,
    parameter NUM_ROWS = 32,
    parameter NUM_COLS = 32,
    parameter IDATA_WIDTH = 16,
    parameter STAGES_NUM = $clog2(NUM_ROWS),
    parameter INPUTS_NUM_INT = 2 ** STAGES_NUM,
    parameter ODATA_WIDTH = IDATA_WIDTH + STAGES_NUM,
    parameter NUM_SUB_MACROS = 4 // Number of cim_sub_macro instances
) (
    input clk,
    input rst_n,
    input [NUM_SUB_MACROS-1:0] en,
    input [NUM_SUB_MACROS*NUM_COLS-1:0] sel_cols,
    input [NUM_SUB_MACROS*NUM_COLS-1:0] write_en,
    input [NUM_SUB_MACROS*NUM_ROWS*DATA_WIDTH-1:0] data_lines,
    input [NUM_SUB_MACROS*NUM_ROWS*DATA_WIDTH-1:0] data_lines_n,
    output [NUM_SUB_MACROS*NUM_COLS*ODATA_WIDTH-1:0] psum_buff_out,
    output [NUM_SUB_MACROS-1:0] psum_data_ready,
    input [NUM_SUB_MACROS-1:0] psum_ack
);

  genvar i;
  generate
    for (i = 0; i < NUM_SUB_MACROS; i = i + 1) begin : sub_macro_gen
      cim_sub_macro #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_ROWS(NUM_ROWS),
        .NUM_COLS(NUM_COLS),
        .IDATA_WIDTH(IDATA_WIDTH)
      ) cim_sub_macro_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(en[i]),
        .sel_cols(sel_cols[i*NUM_COLS +: NUM_COLS]),
        .write_en(write_en[i*NUM_COLS +: NUM_COLS]),
        .data_lines(data_lines[i*NUM_ROWS*DATA_WIDTH +: NUM_ROWS*DATA_WIDTH]),
        .data_lines_n(data_lines_n[i*NUM_ROWS*DATA_WIDTH +: NUM_ROWS*DATA_WIDTH]),
        .psum_buff_out(psum_buff_out[i*NUM_COLS*ODATA_WIDTH +: NUM_COLS*ODATA_WIDTH]),
        .psum_data_ready(psum_data_ready[i]),
        .psum_ack(psum_ack[i])
      );
    end
  endgenerate

endmodule