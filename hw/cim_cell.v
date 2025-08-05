module cim_cell #(
    parameter DATA_WIDTH = 9,
) (
    clk,
    rst_n,

    en,
    sel_cell, // Cell select

    write_en,
    data_line,
    data_line_n,

    mac_result_out,
    out_data_valid, // To column-wise adder tree
);

  input clk;
  input rst_n;
  input sel_cell;

  input en;
  input write_en;

  input [DATA_WIDTH-1 : 0] data_line;
  input [DATA_WIDTH-1 : 0] data_line_n;

  output reg [2*(DATA_WIDTH-1)-1 : 0] mac_result_out; // 16 bit result
  output reg out_data_valid;
  
  wire [DATA_WIDTH-2 : 0] weight;
  wire metadata_bit;
  wire [DATA_WIDTH-2 : 0] activation;
  wire [DATA_WIDTH-1 : 0] read_out;

  cim_mem_cell #(.DATA_WIDTH(DATA_WIDTH))
  cim_mem_cell_inst(
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .write_en(write_en),
    .data_line(data_line),
    .read_out(read_out)
  );

  assign metadata_bit = read_out[0]; // Assign LSB to metadata_bit
  assign weight = read_out[DATA_WIDTH-1:1]; // Assign the rest to weight
  assign activation = metadata_bit == 0 ? data_line[DATA_WIDTH-1:1] : data_line_n[DATA_WIDTH-1:1];

  always @(posedge clk) begin
    if(en == 1) begin
        if (sel_cell == 1) begin
            mac_result_out <= weight * activation;
            out_data_valid <= 1;
        end
        else
            mac_result_out <= {2*(DATA_WIDTH-1){1'b0}};
            out_data_valid <= 0;
    end

    else begin
        mac_result_out <= {2*(DATA_WIDTH-1){1'b0}};
        out_data_valid <= 0;
    end
  end
endmodule