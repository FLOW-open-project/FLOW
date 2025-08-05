module flexcim #(
    parameter NUM_INPUT_DATA = 4, // Number of input elements for MUX
    parameter NUM_SUB_MACROS = 4,
    parameter DATA_WIDTH = 9,
    parameter NUM_ROWS = 32,
    parameter NUM_COLS = 32,
    parameter IDATA_WIDTH = 16,
    parameter ODATA_WIDTH_FINAL = 22,

    parameter STAGES_NUM = $clog2(NUM_ROWS),
    parameter INPUTS_NUM_INT = 2 ** STAGES_NUM,
    parameter ODATA_WIDTH = IDATA_WIDTH + STAGES_NUM

) (
    input clk,
    input rst_n,
    input [NUM_SUB_MACROS-1:0] en,
    input [NUM_ROWS-1 : 0] i_valid,
    input [NUM_ROWS*NUM_SUB_MACROS*NUM_INPUT_DATA*2*DATA_WIDTH-1:0] i_data_bus,
    input [NUM_ROWS*2*NUM_SUB_MACROS*2-1 : 0] i_sparse_select, 
    input [NUM_SUB_MACROS*NUM_COLS-1:0] sel_cols,
    input [NUM_SUB_MACROS*NUM_COLS-1:0] write_en,

    output [NUM_COLS*ODATA_WIDTH_FINAL-1:0] psum_final
);


wire [NUM_ROWS-1 : 0]  o_valid;
wire [NUM_ROWS*NUM_SUB_MACROS*2*DATA_WIDTH-1:0] o_data_bus;
wire [NUM_ROWS*NUM_SUB_MACROS*DATA_WIDTH-1:0] data_lines;
wire [NUM_ROWS*NUM_SUB_MACROS*DATA_WIDTH-1:0] data_lines_n;
wire [NUM_SUB_MACROS*NUM_COLS*ODATA_WIDTH-1:0] psum_buff_out;
wire [NUM_SUB_MACROS-1:0] psum_data_ready;
wire [NUM_SUB_MACROS-1:0] psum_ack;
genvar i;
generate
    for (i = 0; i < NUM_ROWS; i = i + 1) begin : distribution_inst
        distribution #(
            .DATA_WIDTH(2*DATA_WIDTH),
            .NUM_INPUT_DATA(NUM_INPUT_DATA),
            .NUM_SUB_MACROS(NUM_SUB_MACROS)
        ) distribution_inst (
            .clk(clk),
            .rst_n(rst_n),
            .i_valid(i_valid[i]),
            .i_data_bus(i_data_bus[i*NUM_SUB_MACROS*NUM_INPUT_DATA*2*DATA_WIDTH +: NUM_SUB_MACROS*NUM_INPUT_DATA*2*DATA_WIDTH]),
            .i_en(en[0]),
            .i_sparse_select(i_sparse_select[i*2*NUM_SUB_MACROS*2 +: 2*NUM_SUB_MACROS*2]),
            .o_valid(o_valid[i]), // Currently not used since control will handle the flow
            .o_data_bus(o_data_bus[i*NUM_SUB_MACROS*2*DATA_WIDTH +: NUM_SUB_MACROS*2*DATA_WIDTH])
        );
    end
endgenerate

genvar j;
generate
    for (j = 0; j < NUM_ROWS*NUM_SUB_MACROS; j = j + 1) begin : data_line_assignment
        assign data_lines[j*DATA_WIDTH +: DATA_WIDTH] = o_data_bus[j*2*DATA_WIDTH +: DATA_WIDTH];
        assign data_lines_n[j*DATA_WIDTH +: DATA_WIDTH] = o_data_bus[j*2*DATA_WIDTH + DATA_WIDTH +: DATA_WIDTH];
    end
endgenerate

cim_macro #(
    .DATA_WIDTH(DATA_WIDTH),
    .NUM_ROWS(NUM_ROWS),
    .NUM_COLS(NUM_COLS),
    .IDATA_WIDTH(IDATA_WIDTH),
    .NUM_SUB_MACROS(NUM_SUB_MACROS)
) cim_macro_inst (
    .clk(clk),
    .rst_n(rst_n),
    .en(en),
    .sel_cols(sel_cols),
    .write_en(write_en),
    .data_lines(data_lines),
    .data_lines_n(data_lines_n),
    .psum_buff_out(psum_buff_out),
    .psum_data_ready(psum_data_ready),
    .psum_ack(psum_ack)
);

    merging #(
        .NUM_SUB_MACROS(NUM_SUB_MACROS),
        .NUM_COLS(NUM_COLS),
        .ODATA_WIDTH(ODATA_WIDTH),
        .ODATA_WIDTH_FINAL(ODATA_WIDTH_FINAL)
    ) merging_instance (
        .clk(clk),
        .rst_n(rst_n),
        .psum_buff_out(psum_buff_out),
        .psum_data_ready(psum_data_ready),
        .psum_ack(psum_ack),
        .psum_final(psum_final)
    );
endmodule