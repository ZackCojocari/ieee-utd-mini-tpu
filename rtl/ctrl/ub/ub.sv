// ub: the unified buffer.
// Holds the whole activation image (a_img) for a layer and returns one 64-bit word per cycle
// to the array.
//
// Ports
//   name      dir    width  meaning
//   clk       in     1      the one clock
//   rst       in     1      synchronous, active high; zeroes the write pointer
//   wr_rst    in     1      pulse; zero the write pointer (CTRL.UB_WR_RST)
//   wr_valid  in     1      store wr_data at the write pointer, then add one
//   wr_data   in     64     s_axis_tdata
//   rd_addr   in     AW     word address, ub_base + k from layer_fsm
//   rd_data   out    64     a_word to the mxu; valid RD_LAT = 2 cycles after rd_addr
//   AW        param  int    13; the memory holds 2**AW = 8,192 words
//
// Behaviour
//   1. Hold a memory of 8,192 x 64 bits: logic [63:0] mem [0:8191].
//   2. Hold a 14-bit write pointer wp. On rst or wr_rst, wp <= 0.
//   3. On a cycle with wr_valid high (and wr_rst low): mem[wp] <= wr_data; wp <= wp + 1.
//      If wr_rst and wr_valid are high together, wr_rst wins and the word is dropped.
//   4. Every cycle: register mem[rd_addr] into a read register, then into an output register
//      that drives rd_data. That is a registered read plus an output register.
//   5. Writes arrive only before START, in a_img order, so word mt*K + k lands at address
//      mt*K + k. Reads arrive only while BUSY. They never overlap, so there is no
//      read-during-write case to handle.
//
// Timing
//   rd_addr presented in cycle t gives rd_data in cycle t + 2. A new address every cycle.
//   No handshake on either port; the write side is always ready.
//   Reset clears wp only. Memory contents need no reset.
//
// Resource target
//   16 RAMB36E1, no LUT RAM, block RAM output registers on.
//
// Done when
//   tb/ctrl/ub/tb_ub.sv prints PASS, and synthesized alone it shows 16 RAMB36E1.
//
// Spec
//   docs/architecture.md, section 6.3. docs/verification.md, section 6.3.

module ub #(
  parameter int AW = 13
) (
  input  logic          clk,
  input  logic          rst,
  input  logic          wr_rst,
  input  logic          wr_valid,
  input  logic [63:0]   wr_data,
  input  logic [AW-1:0] rd_addr,
  output logic [63:0]   rd_data
);

localparam int depth = 1 << AW;
logic [63:0] mem [0:depth-1];
logic [AW:0] wr_ptr = '0;

always_ff@(posedge clk)begin
  if(rst || wr_rst)begin
    wr_ptr <= '0;
  end
  else if(wr_valid)begin
    wr_ptr <= wr_ptr + 1;
    mem[wr_ptr[AW-1:0]] <= wr_data;
  end
end

logic [63:0] rd_lat;

always_ff@(posedge clk)begin
  rd_lat <= mem[rd_addr];
  rd_data <= rd_lat;
end




endmodule
