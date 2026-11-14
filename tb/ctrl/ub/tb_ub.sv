// tb_ub: unit bench for ub.
//
// DUT          rtl/ctrl/ub/ub.sv, AW = 13
// Reference    a plain array of 8,192 x 64 bits in the bench, written in step with the DUT,
//              and a model write pointer. Every read is checked exactly RD_LAT cycles later.
// Assertions   the model write pointer never passes 8,192. rd_data is not X, RD_LAT cycles
//              after reading an address that was written.
// Must hit     address 0 and 8,191 written and read; reads on consecutive cycles; the same
//              address read twice in a row; wr_rst in the middle of a fill.
// Done when    every task below passes and synthesized alone ub shows 16 RAMB36E1.
// Spec         docs/verification.md, section 6.3. docs/architecture.md, section 6.3.
//
// Each test in the plan is a task below; the initial block runs them in order.
// Bench rules: docs/verification.md, section 2.

module tb_ub;
  import tb_pkg::*;

  localparam int AW     = 13;
  localparam int RD_LAT = 2;

  logic          clk = 1'b0;
  logic          rst;
  logic          wr_rst;
  logic          wr_valid;
  logic [63:0]   wr_data;
  logic [AW-1:0] rd_addr;
  logic [63:0]   rd_data;

  always #5 clk = ~clk;

  ub #(.AW(AW)) dut (.*);
  
  logic [63:0] expected_mem [0:8191];
  logic [63:0] expected_rd_data [RD_LAT];
  logic [AW-1:0] expected_rd_addr [RD_LAT];
  bit rd_valid [RD_LAT];
  int unsigned expected_wr_ptr;
  

  always_ff @(posedge clk) begin
    if (rst || wr_rst) begin
      expected_wr_ptr <= 0;
    end else if (wr_valid) begin
      if (expected_wr_ptr >= 8192) 
        $fatal(1, "ub testbench: write beyond address 8191");
      else begin 

      expected_mem[expected_wr_ptr] <= wr_data;
      expected_wr_ptr <= expected_wr_ptr +1;
      end
    end
  end

  always_ff @(posedge clk) begin
    expected_rd_data[0] <= expected_mem[rd_addr];
    expected_rd_addr[0] <= rd_addr;
    rd_valid[0] <= !rst && (expected_wr_ptr > rd_addr);

    for (int i = 1; i < RD_LAT; i++) begin
      expected_rd_data[i] <= expected_rd_data[i-1];
      expected_rd_addr[i] <= expected_rd_addr[i-1];
      rd_valid[i] <= rd_valid[i-1];
    end 
  end

  always @(negedge clk) begin
    if (rd_valid[RD_LAT-1] && rd_data !== expected_rd_data[RD_LAT-1])
      $fatal(1, "ub read at address %0d: got %h, expected %h",
             expected_rd_addr[RD_LAT-1], rd_data, expected_rd_data[RD_LAT-1]);
  end






  initial begin
    $display("SKIP: not implemented"); $finish;
    void'(get_seed());
    test_fill_and_read_all();
    test_fsm_pattern();
    test_wr_rst_midfill();
    test_addr_every_cycle();
    test_reset_midway();
    test_random();
    finish_pass();
  end

  // 1. Fill all 8,192 words with a value derived from the address, read all back in order; fails on any mismatch.
  task automatic test_fill_and_read_all();
  endtask

  // 2. Read base + k for k = 0..K-1 for several bases and K; fails if any word differs from the model at RD_LAT.
  task automatic test_fsm_pattern();
  endtask

  // 3. wr_rst halfway through a fill, then refill with a second image; fails unless the second image is what reads back.
  task automatic test_wr_rst_midfill();
  endtask

  // 4. A new read address every cycle; fails if any read is lost, repeated or late against RD_LAT.
  task automatic test_addr_every_cycle();
  endtask

  // Rule 5. rst during a fill, then a clean fill from address 0; fails unless the clean image reads back exactly.
  task automatic test_reset_midway();
  endtask

  // Random. 50,000 ops: random data with random wr_valid gaps, then a random read address every cycle; fails on any mismatch.
  task automatic test_random();
  endtask

endmodule
