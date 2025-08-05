module cim_mem_cell #(
    parameter DATA_WIDTH = 9, // One bit for metadata to choose from either of the two data lines Metadata is stored in LSB bit
) (
    clk,
    rst_n,

    en,

    write_en,
    data_line,

    read_out 
);

  input clk;
  input rst_n;

  input en;
  input write_en;

  input [DATA_WIDTH-1 : 0] data_line;

  output reg [DATA_WIDTH-1 : 0] read_out;

  reg [DATA_WIDTH-1 : 0] weight_ram;

  initial begin
    weight_ram = {DATA_WIDTH{1'b0}};
  end

  always @(posedge clk) begin
    if(en == 1 && rst_n == 1) begin
        read_out <= weight_ram;
        if(write_en) begin
            weight_ram <= data_line;
        end
    end
    else
        read_out <= {DATA_WIDTH{1'b0}};
  end
endmodule