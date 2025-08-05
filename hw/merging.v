module merging #(
    parameter NUM_SUB_MACROS = 4,
    parameter NUM_COLS = 32,
    parameter ODATA_WIDTH = 20,
    parameter ODATA_WIDTH_FINAL = 22

) (
    input clk,
    input rst_n,

    input [NUM_SUB_MACROS*NUM_COLS*ODATA_WIDTH-1:0] psum_buff_out,
    input [NUM_SUB_MACROS-1:0] psum_data_ready,
    output reg [NUM_SUB_MACROS-1:0] psum_ack,

    output reg [NUM_COLS*ODATA_WIDTH_FINAL-1:0] psum_final
);

    reg [NUM_SUB_MACROS*NUM_COLS*ODATA_WIDTH-1:0] registered_data;
    reg [NUM_SUB_MACROS-1:0] i_data_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_ack <= 0;
            registered_data <= 0;
        end else begin
            if (psum_data_ready == {NUM_SUB_MACROS{1'b1}}) begin
                psum_ack <= {NUM_SUB_MACROS{1'b1}};
                registered_data <= psum_buff_out;
                i_data_valid <= {NUM_SUB_MACROS{1'b1}};
            end else begin
                psum_ack <= 0;
                i_data_valid <= {NUM_SUB_MACROS{1'b0}};
            end
        end
    end

generate
    genvar col;
    for (col = 0; col < NUM_COLS; col = col + 1) begin : adder_tree_gen
        wire [ODATA_WIDTH_FINAL-1:0] mac_col_result_out;
        wire acc_ready_col;
        
        adder_tree #(
            .INPUTS_NUM(NUM_SUB_MACROS), 
            .IDATA_WIDTH(ODATA_WIDTH)
        ) adder_tree_instance (
            .clk(clk),
            .rst_n(rst_n),
            .i_data_valid(i_data_valid),
            .idata(registered_data[col*ODATA_WIDTH +: ODATA_WIDTH*NUM_SUB_MACROS]),
            .odata(mac_col_result_out),
            .acc_ready(acc_ready_col)
        );

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                psum_final[col*ODATA_WIDTH_FINAL +: ODATA_WIDTH_FINAL] <= 0;
            end 
            
            else if (acc_ready_col) begin
                psum_final[col*ODATA_WIDTH_FINAL +: ODATA_WIDTH_FINAL] <= mac_col_result_out;
            end
        end
    end
endgenerate
endmodule