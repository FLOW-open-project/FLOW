module cim_column #(
    parameter DATA_WIDTH = 9,
    parameter NUM_ROWS = 32,
    parameter IDATA_WIDTH = 16,
    parameter STAGES_NUM = $clog2(NUM_ROWS),
    parameter INPUTS_NUM_INT = 2 ** STAGES_NUM,
    parameter ODATA_WIDTH = IDATA_WIDTH + STAGES_NUM
) (
    clk,
    rst_n,

    en,
    sel_col, // Column select

    write_en,
    data_lines,
    data_lines_n,

    mac_col_result_out,
    acc_ready
);

  input clk;
  input rst_n;
  input sel_col;

  input en;
  input write_en;
  
  input [NUM_ROWS*DATA_WIDTH-1 : 0] data_lines;
  input [NUM_ROWS*DATA_WIDTH-1 : 0] data_lines_n;

  output [ODATA_WIDTH-1 : 0] mac_col_result_out;
  output acc_ready;
  
  wire [NUM_ROWS*IDATA_WIDTH-1:0] idata;
  wire [NUM_ROWS-1:0] i_data_valid;

  genvar row_num;
  generate;
    for( row_num = 0; row_num < NUM_ROWS; row_num++ ) begin
        cim_cell #(.DATA_WIDTH = DATA_WIDTH)
        cim_cell_instance_i_j(
            .clk(clk)
            .rst_n(rst_n),
            .en(en),
            .sel_cell(sel_col),
            .write_en(write_en),
            .data_line(data_lines[row_num*DATA_WIDTH +: DATA_WIDTH]),
            .data_line_n(data_lines[row_num*DATA_WIDTH +: DATA_WIDTH]),
            .mac_result_out(idata[row_num*IDATA_WIDTH +: IDATA_WIDTH]),
            .out_data_valid(i_data_valid[row_num])
        );
    end
  endgenerate


  adder_tree #(.INPUTS_NUM(NUM_ROWS), .IDATA_WIDTH(IDATA_WIDTH))
  adder_tree_instance(
    .clk(clk),
    .rst_n(rst_n)
    .i_data_valid(i_data_valid),
    .idata(idata),
    .odata(mac_col_result_out),
    .acc_ready(acc_ready)
  );


endmodule