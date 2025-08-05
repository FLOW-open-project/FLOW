module cim_sub_macro #(
    parameter DATA_WIDTH = 9,
    parameter NUM_ROWS = 32,
    parameter NUM_COLS = 32,
    parameter IDATA_WIDTH = 16,
    parameter STAGES_NUM = $clog2(NUM_ROWS),
    parameter INPUTS_NUM_INT = 2 ** STAGES_NUM,
    parameter ODATA_WIDTH = IDATA_WIDTH + STAGES_NUM
) (
    clk,
    rst_n,

    en,
    sel_cols, // Column select

    write_en,
    data_lines,
    data_lines_n,

    psum_buff_out,
    psum_data_ready,
    psum_ack

);

  input clk;
  input rst_n;
  input [NUM_COLS-1 : 0] sel_cols;

  input en;
  input [NUM_COLS-1 : 0] write_en;
  
  input [NUM_ROWS*DATA_WIDTH-1 : 0] data_lines;
  input [NUM_ROWS*DATA_WIDTH-1 : 0] data_lines_n;

  output [NUM_COLS*ODATA_WIDTH-1:0] psum_buff_out;
  output reg psum_data_ready;
  input psum_ack;

  wire [NUM_COLS*ODATA_WIDTH-1 : 0] mac_col_result_out;
  reg [NUM_COLS-1 : 0] acc_ready;

  reg [ODATA_WIDTH-1:0] psum_buffer[NUM_COLS-1:0][2:0]; // 3-element buffer for each column
  reg [1:0] psum_write_index[NUM_COLS-1:0]; // 2-bit index for cyclic access
  reg [1:0] psum_read_index[NUM_COLS-1:0]; // 2-bit index for cyclic read access

  genvar col_num;
  generate;
    for( col_num = 0; col_num < NUM_COLS; col_num++ ) begin
        cim_column #(.DATA_WIDTH(DATA_WIDTH), .NUM_ROWS(NUM_ROWS), .IDATA_WIDTH(IDATA_WIDTH))
        cim_column_inst(
          .clk(clk),
          .rst_n(rst_n),
          .en(en),
          .sel_col(sel_cols[col_num]),
          .write_en(write_en[col_num]),
          .data_lines(data_lines),
          .data_lines_n(data_lines_n),
          .mac_col_result_out(mac_col_result_out[col_num*ODATA_WIDTH +: ODATA_WIDTH]),
          .acc_ready(acc_ready[col_num])
        );


    end
  endgenerate


  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Reset logic
      integer i, j;
      for (i = 0; i < NUM_COLS; i = i + 1) begin
        psum_write_index[i] <= 0;
        psum_read_index[i] <= 0;
        for (j = 0; j < 3; j = j + 1) begin
          psum_buffer[i][j] <= 0;
        end
      end
    end else begin
      // Write update logic
      integer k;
      for (k = 0; k < NUM_COLS; k = k + 1) begin
        if (acc_ready[k]) begin
          psum_buffer[k][psum_write_index[k]] <= mac_col_result_out[k*ODATA_WIDTH +: ODATA_WIDTH];
          psum_write_index[k] <= (psum_write_index[k] + 1) % 3;
        end
      end

      // Read index update logic
      if (psum_ack) begin
        integer m;
        for (m = 0; m < NUM_COLS; m = m + 1) begin
          if (psum_read_index[m] != psum_index[m]) begin
            psum_read_index[m] <= psum_read_index[m] + 1;
            psum_data_ready <= 1;
          end

          else
            psum_data_ready <= 0;
        end
      end
    end
  end

  // Assign the current read value to the output
  genvar n;
  generate
    for (n = 0; n < NUM_COLS; n = n + 1) begin
      assign psum_buff_out[n*ODATA_WIDTH +: ODATA_WIDTH] = psum_buffer[n][psum_read_index[n]];
    end
  endgenerate

endmodule