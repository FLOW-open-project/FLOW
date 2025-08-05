module adder_tree #(
  parameter INPUTS_NUM = 32,
  parameter IDATA_WIDTH = 16,

  parameter STAGES_NUM = $clog2(INPUTS_NUM),
  parameter INPUTS_NUM_INT = 2 ** STAGES_NUM,
  parameter ODATA_WIDTH = IDATA_WIDTH + STAGES_NUM
)(
    clk,
    rst_n,
    i_data_valid,
    idata,
    odata,
    acc_ready
);

    input clk;
    input rst_n;
    input [INPUTS_NUM*IDATA_WIDTH-1:0] idata;
    input [INPUTS_NUM-1:0] i_data_valid;
    output [ODATA_WIDTH-1:0] odata;
    output reg acc_ready;

    reg [STAGES_NUM:0][INPUTS_NUM_INT-1:0][ODATA_WIDTH-1:0] data;

// generating tree
genvar stage, adder;
generate
  for( stage = 0; stage <= STAGES_NUM; stage++ ) begin: stage_gen
    if(stage == STAGES_NUM)
        acc_ready <= 1;
    else
        acc_ready <= 0;
    localparam ST_OUT_NUM = INPUTS_NUM_INT >> stage;
    localparam ST_WIDTH = IDATA_WIDTH + stage;

    if( stage == '0 ) begin
      // stege 0 is actually module inputs
      for( adder = 0; adder < ST_OUT_NUM; adder++ ) begin: inputs_gen

        always@(*) begin
          if( adder < INPUTS_NUM ) begin
            if(i_data_valid[adder] == 1) begin
                data[stage][adder][ST_WIDTH-1:0] <= idata[adder*IDATA_WIDTH +: ST_WIDTH];
                data[stage][adder][ODATA_WIDTH-1:ST_WIDTH] <= '0;
            end
          end else begin
            data[stage][adder][ODATA_WIDTH-1:0] <= '0;
          end
        end // always_comb

      end // for
    end else begin
      // all other stages hold adders outputs
      for( adder = 0; adder < ST_OUT_NUM; adder++ ) begin: adder_gen

        //always_comb begin       // is also possible here
        always@(posedge clk) begin
          if( ~rst_n ) begin
            data[stage][adder][ODATA_WIDTH-1:0] <= '0;
          end else begin
            data[stage][adder][ST_WIDTH-1:0] <=
                    data[stage-1][adder*2][(ST_WIDTH-1)-1:0] +
                    data[stage-1][adder*2+1][(ST_WIDTH-1)-1:0];
          end
        end // always
      end // for
    end // if stage
  end // for
endgenerate

assign odata = data[STAGES_NUM][0];

endmodule