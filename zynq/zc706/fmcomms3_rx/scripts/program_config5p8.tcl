# Manual volatile JTAG load. No QSPI programming is performed.
# xsct scripts/program_config5p8.tcl run
set root [file normalize [file join [file dirname [info script]] ..]]
set elf $root/export/key_ctrlll_5p8.elf
set bit $root/export/zc706_top.bit
set psinit $root/export/ps7_init.tcl
foreach file [list $elf $bit $psinit] {
    if {![file exists $file]} {error "Missing prepared artifact: $file"}
}
connect -url tcp:127.0.0.1:3121
after 1500
if {[llength [targets -target-properties -filter {name =~ "xc7z045*"}]] != 1} {
    error "Expected exactly one XC7Z045 target; select the ZC706 cable explicitly if multiple boards are attached"
}
targets -set -filter {name == "APU"}
rst -system
after 1000
targets -set -filter {name =~ "xc7z045*"}
fpga -file $bit
targets -set -filter {name == "APU"}
configparams force-mem-access 1
source $psinit
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*A9*#0"}
dow $elf
configparams force-mem-access 0
if {[lindex $argv 0] eq "run"} {
    con
    puts "CONFIG5P8 application running; UART1 115200 8N1"
} else {
    puts "CONFIG5P8 downloaded and halted; use run argument to start initialization"
}
disconnect
exit
