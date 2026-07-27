# simulation.tcl: run csim and write sim_out/{design}.txt

# 1) Read env vars
foreach v {DESIGN_K DESIGN_RUN DESIGN_START DESIGN_END} {
    if {![info exists ::env($v)]} {
        error "$v not set"
    }
}
set K        $::env(DESIGN_K)
set RUN      $::env(DESIGN_RUN)
set startIdx $::env(DESIGN_START)
set endIdx   $::env(DESIGN_END)
puts "DEBUG: K=$K RUN=$RUN START=$startIdx END=$endIdx"

# 2) Count problems via run1 files
set run1files [lsort -dictionary [glob -nocomplain *_design_run1.cpp]]
if {[llength $run1files]==0} { error "No *_design_run1.cpp files" }
set totalProb [llength $run1files]
if {$endIdx > $totalProb} { set endIdx $totalProb }

puts "Pass@${K}: run ${RUN}, problems ${startIdx}–${endIdx}/${totalProb}"

# 3) Write sim result to an isolated per-design file (safe for parallel runs)
file mkdir sim_out
proc write_sim_result {designNm compStatus simStatus message} {
    set fh [open "sim_out/${designNm}.txt" w]
    puts $fh $compStatus
    puts $fh $simStatus
    puts $fh $message
    close $fh
}

# 4) Simulate this run slice
set allRunFiles [lsort -dictionary [glob -nocomplain *_design_run${RUN}.cpp]]
set subRuns     [lrange $allRunFiles [expr {$startIdx-1}] [expr {$endIdx-1}]]

foreach srcFile $subRuns {
    regexp {^(.+?)_design_run([0-9]+)\.cpp$} $srcFile -> prefix runNo
    set tbFile   "${prefix}_tb.cpp"
    set designNm "${prefix}_run${runNo}"

    if {![file exists $tbFile]} {
        write_sim_result $designNm FAIL FAIL "Missing testbench"
        continue
    }

    file delete -force "$designNm"
    open_project   -reset "$designNm"
    set_top        TopModule
    add_files      $srcFile
    add_files -tb  $tbFile
    open_solution  -reset simulation
    create_clock   -period 10 -name default

    # On Vivado 2018.3 Windows, csim_design fails due to /dev/null in Makefile.rules.
    # Workaround: run csim_design -setup (may fail but generates files), fix Makefile.rules,
    # then run make manually.
    catch { csim_design -setup }
    set simOutput ""
    set compRc 1
    # Fix /dev/null → NUL in Makefile.rules
    set rulesFile [file normalize "${designNm}/simulation/csim/build/Makefile.rules"]
    if {[file exists $rulesFile]} {
        set fh [open $rulesFile r]
        set content [read $fh]
        close $fh
        regsub -all {/dev/null} $content "NUL" content
        set fh [open $rulesFile w]
        puts -nonewline $fh $content
        close $fh
        # Run make with Vivado's msys64 in PATH
        set vivadoRoot "D:/Xilinx/Vivado/2018.3"
        set msysBin "${vivadoRoot}/msys64/mingw64/bin"
        set mingwBin "${vivadoRoot}/msys64/usr/bin"
        set ::env(PATH) "${msysBin};${mingwBin};${vivadoRoot}/bin;$::env(PATH)"
        set buildDir [file normalize "${designNm}/simulation/csim/build"]
        set makePath "${msysBin}/make.exe"
        if {[file exists $makePath]} {
            set compRc [catch { exec "${makePath}" -C "${buildDir}" -f csim.mk 2>&1 } simOutput]
            if {$compRc == 0} {
                # Run the compiled executable with proper PATH via a wrapper batch file
                set exeFile "${buildDir}/csim.exe"
                if {[file exists $exeFile]} {
                    # Create a wrapper batch file that sets PATH and runs csim.exe
                    set wrapperBat "${buildDir}/run_csim.bat"
                    set fh [open $wrapperBat w]
                    puts $fh "@echo off"
                    puts $fh "set PATH=${msysBin};${mingwBin};${vivadoRoot}/bin;%PATH%"
                    puts $fh "\"${exeFile}\""
                    close $fh
                    # Use open with pipe to capture output
                    set pipeFd [open "|cmd /c \"${wrapperBat}\"" r]
                    set simOutput [read $pipeFd]
                    set simRc [catch { close $pipeFd }]
                    set compRc $simRc
                }
            }
        }
    }

    if {$compRc != 0} {
        write_sim_result $designNm FAIL FAIL "csim_design failed: compilation error(s)"
    } else {
        # Check simOutput for pass/fail marker
        if {$simOutput eq ""} {
            # Try reading the log file as fallback
            set logFile [file normalize "${designNm}/simulation/csim/report/TopModule_csim.log"]
            if {![file exists $logFile]} {
                set logFile [file normalize "${designNm}/simulation/csim/report/Topmodule_csim.log"]
            }
            if {[file exists $logFile]} {
                set fh2 [open $logFile r]
                set simOutput [read $fh2]
                close $fh2
            }
        }

        if {[string match "*Test Passed*" $simOutput]} {
            write_sim_result $designNm PASS PASS "Test Passed"
        } elseif {[string match "*Test Failed*" $simOutput]} {
            write_sim_result $designNm PASS FAIL "Test Failed"
        } else {
            write_sim_result $designNm PASS FAIL "No pass/fail marker in csim output"
        }
    }

    catch { close_project }
}

quit
